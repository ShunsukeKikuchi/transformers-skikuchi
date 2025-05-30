# coding=utf-8
# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Dia model."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import RMSNorm

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation.utils import GenerationConfig, GenerationMixin
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    is_torch_flex_attn_available,
    is_torchdynamo_compiling,
    logging,
)
from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from .generation_dia import DiaGenerationMixin


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import BlockMask, make_flex_block_causal_mask


logger = logging.get_logger(__name__)


class DiaPreTrainedModel(PreTrainedModel):
    config_class = DiaConfig
    base_model_prefix = "model"
    main_input_name = "encoder_input_ids"  # TODO: change this?
    supports_gradient_checkpointing = True
    _no_split_modules = ["DiaEncoderLayer", "DiaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = getattr(self.config, "init_std", 0.2)
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    # TODO: refactor masks when merging with main --> new masking for causal etc

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Optional[Union[torch.Tensor, "BlockMask"]],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            # Other attention flavors support in-built causal (when `mask is None`)
            # while we need to create our specific block mask regardless
            elif attention_mask is None:
                attention_mask = make_flex_block_causal_mask(
                    torch.ones(
                        size=(input_tensor.shape[0], input_tensor.shape[1]),
                        device=attention_mask.device,
                    )
                )
            return attention_mask

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_cross_attn_mask
    def _update_cross_attn_mask(
        self,
        encoder_hidden_states: Union[torch.Tensor, None],
        encoder_attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
    ):
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(encoder_attention_mask, torch.Tensor):
                    encoder_attention_mask = make_flex_block_causal_mask(
                        encoder_attention_mask,
                        query_length=input_shape[-1],
                        is_causal=False,
                    )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        return encoder_attention_mask


def apply_rotary_pos_emb(
    tensor: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = position_embeddings[0]
    sin = position_embeddings[1]
    first_half, second_half = torch.chunk(tensor.to(torch.float32), 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat((first_part.to(tensor.dtype), second_part.to(tensor.dtype)), dim=-1)


# TODO: refactor RoPE to transformers rope?
class DiaRotaryEmbedding(nn.Module):
    def __init__(self, config: Union[DiaEncoderConfig, DiaDecoderConfig], device: Optional[torch.device] = None):
        super().__init__()
        self.embedding_dims = config.head_dim
        self.min_timescale = config.rope_min_timescale
        self.max_timescale = config.rope_max_timescale

        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / self.embedding_dims
        freqs = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids_expanded = position_ids[:, :, None, None].float().repeat(x.shape[0], 1, 1, 1)
        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / self.embedding_dims
        freqs = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)

        full_freqs = position_ids_expanded.float() / freqs.to(position_ids_expanded.device)
        cos, sin = full_freqs.cos(), full_freqs.sin()
        return cos, sin


# TODO: copy from llama
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, query.shape[1] // key.shape[1])
    value_states = repeat_kv(value, query.shape[1] // key.shape[1])

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DiaSelfAttention(nn.Module):  # Modular : LlamaAttentions
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[DiaEncoderConfig, DiaDecoderConfig], layer_idx: int, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads or self.num_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.layer_idx = layer_idx
        self.scaling = 1
        self.is_causal = is_causal
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = apply_rotary_pos_emb(query_states, position_embeddings, -2).transpose(1, 2)
        key_states = apply_rotary_pos_emb(key_states, position_embeddings, -2).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_positions": cache_position}
            )

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = self.config.cross_num_attention_heads
        self.num_key_value_heads = self.config.cross_num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.cross_head_dim
        self.layer_idx = layer_idx
        self.scaling = 1
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.num_key_value_heads * self.head_dim, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.num_key_value_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # self.num_key_value_heads = 16 TODO

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if cross_attention_states is not None:
            cross_shape = (*cross_attention_states.shape[:-1], -1, self.head_dim)
            key_states = self.k_proj(cross_attention_states).view(cross_shape).transpose(1, 2)
            value_states = self.v_proj(cross_attention_states).view(cross_shape).transpose(1, 2)
            if past_key_values is not None:
                # TODO: mark layer + general cache fixing + I dont think we need the cache positions anymore
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_positions": cache_position}
                )
        elif past_key_values is not None:  # not prefill, make it compile compatible
            key_states = past_key_values.key_cache[self.layer_idx]  # ty: ignore[unresolved-attribute]
            value_states = past_key_values.value_cache[self.layer_idx]  # ty: ignore[unresolved-attribute]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape((*input_shape, -1)).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DiaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaEncoderConfig, layer_idx: int):
        super().__init__()
        self.pre_sa_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attention = DiaSelfAttention(config, layer_idx)
        self.post_sa_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attention(
            hidden_states=normed_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        normed_states = self.post_sa_norm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = residual + mlp_out

        return hidden_states, self_attn_weights


class DiaEncoder(DiaPreTrainedModel):
    def __init__(self, config: DiaEncoderConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DiaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.float32,
        )
        self.rotary_embeddings = DiaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        if cache_position is None:
            cache_position = torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :]

        hidden_states = self.embedding(input_ids)
        position_embeddings = self.rotary_embeddings(hidden_states, cache_position)

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # TODO: gradient ckpting
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            encoder_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class DiaMLP(nn.Module):  # Modular GlmMLP
    def __init__(self, config):
        super().__init__()

        self.config = config
        # TODO gate_up_proj and down_proj name
        self.wi_fused = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.wi_fused(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.wo(up_states)


class DiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.dropout = config.dropout
        self.self_attention = DiaSelfAttention(config, layer_idx)
        self.cross_attention = DiaCrossAttention(config, layer_idx)
        self.pre_sa_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.pre_ca_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        cache_position: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO: fix caching

        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attention(
            hidden_states=normed_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values.self_attention_cache if past_key_values is not None else None,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )

        # TODO: dropout isnt used anywhere
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ca_norm(hidden_states)
        cross_states, cross_attn_weights = self.cross_attention(
            hidden_states=hidden_states,
            cross_attention_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=encoder_attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values.cross_attention_cache if past_key_values is not None else None,
            output_attentions=output_attentions,
        )

        cross_states = nn.functional.dropout(cross_states, p=self.dropout, training=self.training)
        hidden_states = residual + cross_states

        residual = hidden_states
        x_norm = self.pre_mlp_norm(hidden_states)
        mlp_out = self.mlp(x_norm)
        hidden_states = residual + mlp_out

        return hidden_states, self_attn_weights, cross_attn_weights


class DiaMultiChannelEmbed(nn.Module):
    # TODO: fix description
    """In order to efficiently compute the audio embedding from the 9 different channels
    we vectorize the embedding process by using a single embedding layer, and an offset.
    Example:
    - num_embeds = 3
    - vocab_size = 8
    - num_chanels = 4
    We would have offsets = [0, 256, 512]
    If audio_codes = [0, 1, 2, 3], [1, 3, 4, 7], [5, 6, 7, 8]
    then tokens = audio_codes + offsets
                = [0, 1, 2, 3, 256, 259, 260, 263, 517, 5128, 519, 520]
    This allows us to use a single embedding layer for all channels.
    """

    def __init__(self, config: DiaDecoderConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size * config.num_channels, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        offsets = torch.arange(config.num_channels, dtype=torch.long) * config.vocab_size  # (C,)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        tokens = (audio_codes + self.offsets.to(audio_codes.device)).squeeze(1)
        embeds = self.embed(tokens).view(tokens.shape[0], audio_codes.shape[1], -1, self.hidden_size)
        return embeds.sum(dim=2)


class DiaDecoder(DiaPreTrainedModel):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaDecoderConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        self.vocab_size = config.vocab_size
        self.embeddings = DiaMultiChannelEmbed(config)
        self.rotary_embeddings = DiaRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DiaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.logits_dense = nn.Linear(config.hidden_size, (self.num_channels * self.vocab_size), bias=False)

    def forward(
        self,
        audio_codes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        cross_cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # if use_cache and past_key_values is None:
        if past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        batch_size, seq_length = audio_codes.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=audio_codes.device
            )[None, :]
        if cross_cache_position is None:
            cross_cache_position = torch.arange(encoder_hidden_states.shape[1], device=encoder_hidden_states.device)[
                None, :
            ]

        # RoPE
        hidden_states = self.embeddings(audio_codes)
        position_embeddings = self.rotary_embeddings(hidden_states, cache_position)

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=audio_codes.device)

        # TODO: update to new mask function
        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )

        attention_mask = self._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            self_attn_cache,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_states.shape[:2],
            hidden_states,
        )

        all_self_attn_weights = () if output_attentions else None
        all_cross_attn_weights = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            hidden_states, self_attn_weights, cross_attn_weights = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            if output_attentions:
                all_self_attn_weights = all_self_attn_weights + (self_attn_weights,)
                all_cross_attn_weights = all_cross_attn_weights + (cross_attn_weights,)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        last_hidden_states = self.logits_dense(hidden_states).view(-1, self.num_channels, self.vocab_size)
        if not return_dict:
            return (
                last_hidden_states.to(torch.float32),
                all_hidden_states,
                all_self_attn_weights,
                all_cross_attn_weights,
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attn_weights,
            cross_attentions=all_cross_attn_weights,
        )


class DiaModel(DiaGenerationMixin, DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.encoder = DiaEncoder(config.encoder_config)
        self.decoder = DiaDecoder(config.decoder_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def forward(
        self,
        # TODO: rename back to input_ids and attention_mask
        audio_codes: Optional[Tuple[torch.FloatTensor]] = None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # TODO: return dict
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        if encoder_input_ids is None and encoder_outputs is None:
            raise ValueError(
                "You should either provide text ids or the cached text encodings. Neither has been found."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO: CFG should be possible with logits processor --> move the CFG steps out to prepare inputs
        # batch size becomes 2 * batch_size using CFG (uncoditioned == 0 and conditioned input == text)
        if encoder_input_ids is not None:
            encoder_input_ids = encoder_input_ids[:, None, :]
            unconditioned_encoder_input_ids = torch.zeros_like(encoder_input_ids)
            encoder_input_ids = torch.stack([unconditioned_encoder_input_ids, encoder_input_ids], dim=1).view(
                -1, encoder_input_ids.shape[-1]
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.repeat_interleave(2, dim=0)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                encoder_input_ids,
                attention_mask=encoder_attention_mask,
                cache_position=encoder_cache_position,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Base TTS starts here
        if audio_codes is None:
            # (2*bsz, 1, channel)
            audio_codes = torch.full((encoder_outputs[0].shape[0], 1, 9), 1026, device=self.device)

        decoder_outputs = self.decoder(
            audio_codes=audio_codes,
            attention_mask=None,  # TODO: if we prefix audio we will need a mask when left padding
            cache_position=cache_position,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            cross_cache_position=encoder_cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs,
            # past_key_values=decoder_outputs.past_key_values,
            # decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs[0],
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )


class DiaForConditionalGeneration(GenerationMixin, DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.model = DiaModel(config)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )

        # encoder part
        _encoder_length = self.config.encoder_config.max_length
        _encoder_pad = 0
        encoder_input_ids: Optional[torch.Tensor] = model_kwargs.get("encoder_input_ids", None)
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids must be provided")
        if encoder_input_ids.ndim == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        if encoder_input_ids.ndim != 2:
            raise ValueError("encoder_input_ids must be a 1D or 2D tensor")
        if encoder_input_ids.shape[-1] == _encoder_length:
            raise ValueError(f"encoder_input_ids length must be {_encoder_length} for generation")

        batch_size = encoder_input_ids.shape[0]

        encoder_padding_mask = (encoder_input_ids != _encoder_pad).to(self.device).repeat_interleave(2, dim=0)
        model_kwargs["encoder_attention_mask"] = (
            encoder_padding_mask.unsqueeze(2) & encoder_padding_mask.unsqueeze(1)
        ).unsqueeze(1)

        encoder_uncond_input_ids = torch.zeros_like(encoder_input_ids, dtype=torch.long, device=self.device)
        encoder_input_ids = torch.stack([encoder_uncond_input_ids, encoder_input_ids], dim=1).view(2 * batch_size, -1)

        model_kwargs["encoder_input_ids"] = encoder_input_ids

        # decoder part
        decoder_padding_mask = torch.ones((2 * batch_size, 1), dtype=torch.bool, device=self.device)
        model_kwargs["decoder_attention_mask"] = (
            decoder_padding_mask.unsqueeze(2) & encoder_padding_mask.unsqueeze(1)
        ).unsqueeze(1)

        # decoder eos stopping criteria
        _eos = self.config.eos_token_id or generation_config.eos_token_id
        _pad = self.config.pad_token_id or generation_config.pad_token_id
        _channel = self.config.decoder_config.num_channels
        generation_config._eos_token_tensor = torch.tensor([_pad for _ in range(_channel - 1)] + [_eos])

        # cfg scale
        if model_kwargs.get("cfg_scale", None) is None:
            model_kwargs["cfg_scale"] = 3.0

        # cfg filter top k
        if model_kwargs.get("cfg_filter_top_k", None) is None:
            model_kwargs["cfg_filter_top_k"] = 50

        # audio eos value
        model_kwargs["audio_eos_value"] = _eos
        model_kwargs["audio_pad_value"] = _pad

        # delay pattern
        delay_pattern_list = getattr(self.config, "delay_pattern", None)
        model_kwargs["delay_pattern"] = torch.tensor(delay_pattern_list, device=self.device, dtype=torch.long)
        model_kwargs["max_delay_pattern"] = model_kwargs["delay_pattern"].max().item()
        model_kwargs["num_channels"] = self.config.decoder_config.num_channels

        # eos countdown
        model_kwargs["eos_countdown"] = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        # step
        model_kwargs["step"] = 0
        if model_kwargs.get("max_step", None) is None:
            model_kwargs["max_step"] = self.config.decoder_config.max_length

        return generation_config, model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: Seq2SeqLMOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        eos_countdown = model_kwargs["eos_countdown"]
        audio_eos_value = model_kwargs["audio_eos_value"]
        max_delay_pattern = model_kwargs["max_delay_pattern"]
        step = model_kwargs["step"]
        max_step = model_kwargs["max_step"]

        last_generated_tokens = torch.argmax(outputs.logits, dim=-1)  # Shape [B_orig]

        eos_start_mask = last_generated_tokens == audio_eos_value
        eos_start_mask |= step - max_delay_pattern == max_step
        eos_start_mask &= eos_countdown < 0

        eos_countdown[eos_start_mask] = max_delay_pattern
        eos_countdown[eos_countdown > 0] -= 1

        model_kwargs["step"] = step + 1
        model_kwargs["eos_countdown"] = eos_countdown
        model_kwargs["encoder_outputs"] = outputs.encoder_last_hidden_state

        return model_kwargs

    def forward(
        self,
        encoder_attention_mask: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        eos_countdown: torch.Tensor,
        delay_pattern: torch.Tensor,
        max_delay_pattern: int,
        num_channels: int,
        encoder_input_ids: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cfg_scale: float = 3.0,
        cfg_filter_top_k: int = 50,
        audio_eos_value: int = 1024,
        audio_pad_value: int = 1025,
        step: int = 0,
        max_step: int = 100,
    ) -> Seq2SeqLMOutput:
        """
        Forward method for DiaForConditionalGeneration, following WhisperForConditionalGeneration style.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        lm_logits: torch.Tensor = outputs.last_hidden_state

        loss = None
        # TODO: add loss
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     labels = labels.to(lm_logits.device)  # ty: ignore[invalid-assignment]
        #     loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        # cfg
        logits_last = lm_logits[:, -1].view(lm_logits.shape[0] // 2, 2, *lm_logits.shape[1:])
        uncond_logits = logits_last[:, 0, :]
        cond_logits = logits_last[:, 1, :]
        logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)

        # cfg filter top k
        _, top_k_indices = torch.topk(logits, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices, value=False)
        logits = logits.masked_fill(mask, -torch.inf)

        # EOS filter, this ensures:
        # 1. only channel 0 can generate EOS
        # 2. other channels cannot generate EOS, PAD, BOS
        # 3. if channel 0 has EOS with highest logit, it will be the only candidate
        # 4. if channel 0 has EOS not with highest logit, it will be suppressed
        logits[:, :, audio_eos_value + 1 :] = torch.full_like(
            logits[:, :, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )
        logits[:, 1:, audio_eos_value:] = torch.full_like(
            logits[:, 1:, audio_eos_value:],
            fill_value=-torch.inf,
        )
        logits[:, 0, audio_eos_value] *= torch.tensor(0.8, device=self.device)
        logits_flat = logits.view(-1, logits.shape[-1])

        top_logit_indices = torch.argmax(logits_flat, dim=-1)
        eos_not_highest_mask = top_logit_indices != audio_eos_value
        mask_eos_unless_highest = torch.zeros_like(logits_flat, dtype=torch.bool)
        mask_eos_unless_highest[eos_not_highest_mask, audio_eos_value] = True
        logits_flat = logits_flat.masked_fill(mask_eos_unless_highest, -torch.inf)
        eos_highest_mask = top_logit_indices == audio_eos_value
        mask_eos_highest = torch.zeros_like(logits_flat, dtype=torch.bool)
        mask_eos_highest[eos_highest_mask, :audio_eos_value] = True
        logits_flat = logits_flat.masked_fill(mask_eos_highest, -torch.inf)

        logits = logits_flat.view(logits.shape)

        # EOS countdown and delay pattern application
        active_countdown_mask_Bx = eos_countdown > 0

        if active_countdown_mask_Bx.any():
            # Logits for active items: [num_active, C, V]
            logits_active = logits[active_countdown_mask_Bx]
            # Countdown values for active items: [num_active]
            eos_countdown_active_Bx = eos_countdown[active_countdown_mask_Bx]
            # Delay steps for active items: [num_active]
            current_delay_steps_active_Bx = max_delay_pattern - eos_countdown_active_Bx

            # Expand for comparison with delay_pattern: [num_active, C]
            current_delay_steps_active_BxC = current_delay_steps_active_Bx.unsqueeze(1).expand(-1, num_channels)
            delay_pattern_Cx = delay_pattern.unsqueeze(0).expand(logits_active.shape[0], -1)  # [num_active, C]

            # Mask for forcing EOS: [num_active, C]
            force_eos_mask_BxC = current_delay_steps_active_BxC == delay_pattern_Cx
            # Mask for forcing PAD: [num_active, C]
            force_pad_mask_BxC = current_delay_steps_active_BxC > delay_pattern_Cx

            # Efficiently apply forced EOS and PAD logits
            vocab_size = logits_active.shape[-1]  # logits_active is logits[active_countdown_mask_Bx]

            # Create template rows for forced EOS and PAD
            eos_row = torch.full((vocab_size,), -torch.inf, device=logits.device, dtype=logits_active.dtype)
            eos_row[audio_eos_value] = 0.0
            pad_row = torch.full((vocab_size,), -torch.inf, device=logits.device, dtype=logits_active.dtype)
            pad_row[audio_pad_value] = 0.0

            # Clone the active slice to modify it
            final_modified_slice = logits_active.clone()
            final_modified_slice[force_eos_mask_BxC] = eos_row
            final_modified_slice[force_pad_mask_BxC] = pad_row

            # Update the original logits tensor with the modified slice
            logits[active_countdown_mask_Bx] = final_modified_slice

        if not return_dict:
            # When not returning a dict, the first element is usually logits, then past_kv, then other outputs.
            # The GenerationMixin expects the logits it will sample from as the first element.
            output = (logits,) + outputs[1:]  # Ensure correct logits are passed
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,  # Pass channel 0's logits for GenerationMixin
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


__all__ = [
    "DiaModel",
    "DiaPreTrainedModel",
    "DiaForConditionalGeneration",
]
