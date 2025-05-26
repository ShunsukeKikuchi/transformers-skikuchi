# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Moshi ASR model."""

import inspect
import unittest

import pytest

from transformers import MoshiAsrConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MoshiAsrForConditionalGeneration,
        MoshiAsrModel,
    )


class MoshiAsrModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        input_values_length=448,  # gives 7 audio tokens with the given codec config
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        codebook_vocab_size=2049,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=None,
        max_position_embeddings=512,
        rope_theta=10000.0,
        hidden_act="silu",
        head_dim=None,
        initializer_range=0.02,
        use_cache=True,
        sliding_window=512,
        attention_dropout=0.1,
        ffn_dim=38,
        rms_norm_eps=1e-6,
        num_codebooks=8,
        frame_size=64,
        delay_in_tokens=5,
        audio_bos_token_id=2048,
        audio_pad_token_id=2048,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        codec_config={
            "model_type": "mimi",
            "num_quantizers": 8,
            "audio_channels": 1,
            "chunk_in_sec": None,
            "hidden_size": 16,
            "num_filters": 8,
            "num_residual_layers": 1,
            "upsampling_ratios": [8, 4],
            "codebook_size": 16,
            "vector_quantization_hidden_dimension": 16,
            "upsample_groups": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "sliding_window": 4,
            "codebook_dim": 16,
            "use_cache": False,
        },
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.codebook_vocab_size = codebook_vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.ffn_dim = ffn_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_codebooks = num_codebooks
        self.frame_size = frame_size
        self.delay_in_tokens = delay_in_tokens
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.codec_config = codec_config
        self.scope = scope
        self.input_values_length = input_values_length

    def get_config(self):
        return MoshiAsrConfig(
            codebook_vocab_size=self.codebook_vocab_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            head_dim=self.head_dim,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            sliding_window=self.sliding_window,
            attention_dropout=self.attention_dropout,
            ffn_dim=self.ffn_dim,
            rms_norm_eps=self.rms_norm_eps,
            num_codebooks=self.num_codebooks,
            frame_size=self.frame_size,
            delay_in_tokens=self.delay_in_tokens,
            audio_bos_token_id=self.audio_bos_token_id,
            audio_pad_token_id=self.audio_pad_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            codec_config=self.codec_config,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = MoshiAsrModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs(self):
        config = self.get_config()

        text_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        codebook_input_ids = (
            ids_tensor([self.batch_size, self.seq_length, self.num_codebooks], self.codebook_vocab_size - 1) + 1
        )

        input_ids = torch.cat([text_input_ids.unsqueeze(2), codebook_input_ids], dim=2)
        attention_mask = text_input_ids.ne(1).to(torch_device)

        return config, input_ids, attention_mask

    def prepare_config_and_inputs_generate(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        input_values = floats_tensor([self.batch_size, 1, self.input_values_length])
        # TODO: @eustlb, padding_mask ???
        attention_mask = input_ids.ne(1).to(torch_device)

        return config, input_ids, input_values, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common_generate(self):
        config_and_inputs = self.prepare_config_and_inputs_generate()
        (
            config,
            input_ids,
            input_values,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class MoshiAsrModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            MoshiAsrModel,
            MoshiAsrForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MoshiAsrModel,
            "automatic-speech-recognition": MoshiAsrForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = MoshiAsrForConditionalGeneration if is_torch_available() else None

    def setUp(self):
        self.model_tester = MoshiAsrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MoshiAsrConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels)

        return inputs_dict

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        prepare_config_and_inputs_for_common = self.model_tester.prepare_config_and_inputs_for_common
        self.model_tester.prepare_config_and_inputs_for_common = (
            self.model_tester.prepare_config_and_inputs_for_common_generate
        )
        config, filtered_inputs_dict = super().prepare_config_and_inputs_for_generate()
        self.model_tester.prepare_config_and_inputs_for_common = prepare_config_and_inputs_for_common
        return config, filtered_inputs_dict

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_tie_model_weights(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_resize_embeddings_untied(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_resize_tokens_embeddings(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_tied_weights_keys(self):
        pass

    @pytest.mark.skip(reason="Does not apply to Moshi ASR that requires input_values.")
    def test_generate_without_input_ids(self):
        pass

    def test_initialization(self):
        """
        Overrides [ModelTesterMixin.test_initialization] because of specificities of Mimi codec model.
        See https://github.com/huggingface/transformers/blob/1077603410cd73ba71d64a522033574d66d64b55/tests/models/mimi/test_modeling_mimi.py#L384-L397
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv", "input_proj", "output_proj"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict().keys()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32, *input_ids.shape[2:])
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat(
                (torch.zeros(pad_size[:2], dtype=input_ids.dtype, device=torch_device), attention_mask), dim=1
            )
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)

    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt", "mllama"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            _, inputs = self.model_tester.prepare_config_and_inputs_for_common_generate()
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=4)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]
            if config.is_encoder_decoder:
                inputs["decoder_input_ids"] = outputs_cached.sequences
                if "decoder_attention_mask" in inputs:
                    inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                        inputs["decoder_attention_mask"],
                        (0, new_attention_len - inputs["decoder_attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            else:
                inputs["input_ids"] = outputs_cached.sequences
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.nn.functional.pad(
                        inputs["attention_mask"],
                        (0, new_attention_len - inputs["attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self._check_similar_generate_outputs(outputs, outputs_cached)
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )


# @require_torch_accelerator
# class MoshiAsrIntegrationTest(unittest.TestCase):
#     def tearDown(self):
#         # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
#         # some memory allocated in the cache, which means some object is not being released properly. This causes some
#         # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
#         # Investigate the root cause.
#         cleanup(torch_device, gc_collect=False)

#     @slow
#     @require_read_token
#     def test_llama_3_1_hard(self):
#         """
#         An integration test for llama 3.1. It tests against a long output to ensure the subtle numerical differences
#         from llama 3.1.'s RoPE can be detected
#         """
#         # diff on `EXPECTED_TEXT`:
#         # 2024-08-26: updating from torch 2.3.1 to 2.4.0 slightly changes the results.
#         EXPECTED_TEXT = (
#             "Tell me about the french revolution. The french revolution was a period of radical political and social "
#             "upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked "
#             "by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the "
#             "First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative "
#             "assembly that had not met since 1614. The Third Estate, which represented the common people, "
#             "demanded greater representation and eventually broke away to form the National Assembly. This marked "
#             "the beginning of the end of the absolute monarchy and the rise of the middle class.\n"
#         )

#         tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-MoshiAsr-3.1-8B-Instruct")
#         model = MoshiAsrForCausalLM.from_pretrained(
#             "meta-llama/Meta-MoshiAsr-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16
#         )
#         input_text = ["Tell me about the french revolution."]
#         model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

#         generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
#         generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         self.assertEqual(generated_text, EXPECTED_TEXT)

#     @slow
#     @require_read_token
#     def test_model_7b_logits_bf16(self):
#         input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

#         model = MoshiAsrForCausalLM.from_pretrained(
#             "meta-llama/MoshiAsr-2-7b-hf", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
#         )

#         with torch.no_grad():
#             out = model(torch.tensor([input_ids]).to(torch_device))
#         # Expected mean on dim = -1

#         # fmt: off
#         expected_means = Expectations(
#             {
#             ("xpu", 3): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]]),
#             ("cuda", 7): torch.tensor([[-6.5061, -4.1147, -4.9669, -3.2038, 0.8069, -2.9694, 1.2864, -3.3786]]),
#             ("cuda", 8): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]])
#          })

#         expected_mean = expected_means.get_expectation()
#         self.assertTrue(
#             torch.allclose(
#                 expected_mean.to(torch_device),
#                 out.logits.float().mean(-1),
#                 atol=1e-2,
#                 rtol=1e-2
#             )
#         )

#         # slicing logits[0, 0, 0:15]
#         expected_slices = Expectations(
#             {
#             ("xpu", 3): torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]]),
#             ("cuda", 7): torch.tensor([[-12.5000, -7.0625, -0.6289, -7.8750, -6.9688, -7.8125, -6.4688, -7.4375, -7.6875, -6.9375, -6.0312, -7.0000, -1.8594, 1.8438, -8.5000]]),
#             ("cuda", 8): torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]])
#         })
#         # fmt: on
#         expected_slice = expected_slices.get_expectation()
#         self.assertTrue(
#             torch.allclose(
#                 expected_slice.to(torch_device),
#                 out.logits[0, 0, :15].float(),
#                 atol=1e-2,
#                 rtol=1e-2,
#             )
#         )

#     @slow
#     @require_read_token
#     def test_model_7b_logits(self):
#         input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

#         model = MoshiAsrForCausalLM.from_pretrained(
#             "meta-llama/MoshiAsr-2-7b-hf", device_map="auto", torch_dtype=torch.float16
#         )

#         with torch.no_grad():
#             out = model(torch.tensor([input_ids]).to(torch_device))

#         # fmt: off
#         # Expected mean on dim = -1
#         expected_means = Expectations(
#           {
#             ("xpu", 3): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
#             ("cuda", 7): torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
#             ("cuda", 8): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
#         })

#         expected_mean = expected_means.get_expectation()
#         self.assertTrue(
#             torch.allclose(
#                 expected_mean.to(torch_device),
#                 out.logits.float().mean(-1),
#                 atol=1e-2,
#                 rtol=1e-2
#             )
#         )

#         # slicing logits[0, 0, 0:15]
#         expected_slices = Expectations(
#             {
#               ("xpu", 3): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328]),
#               ("cuda", 7): torch.tensor([-12.8125, -7.3359, -0.4846, -8.0234, -7.2383, -7.9922, -6.4805, -7.7344, -7.8125, -7.0078, -6.1797, -7.1094, -1.8633, 1.9736, -8.6016]),
#               ("cuda", 8): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328])
#         })
#         # fmt: on

#         expected_slice = expected_slices.get_expectation()
#         self.assertTrue(
#             torch.allclose(
#                 expected_slice.to(torch_device),
#                 out.logits[0, 0, :15].float(),
#                 atol=1e-2,
#                 rtol=1e-2,
#             )
#         )

#     @slow
#     def test_model_7b_dola_generation(self):
#         # ground truth text generated with dola_layers="low", repetition_penalty=1.2
#         EXPECTED_TEXT_COMPLETION = (
#             "Simply put, the theory of relativity states that 1) time and space are relative, and 2) the laws of "
#             "physics are the same for all observers in uniform motion relative to one another.\n\nThe theory of "
#             "relativity was developed by Albert Einstein in the early 20th century, and it revolutionized our "
#             "understanding of space and time."
#         )
#         prompt = "Simply put, the theory of relativity states that "
#         tokenizer = MoshiAsrTokenizer.from_pretrained("meta-llama/MoshiAsr-2-7b-chat-hf")
#         model = MoshiAsrForCausalLM.from_pretrained(
#             "meta-llama/MoshiAsr-2-7b-chat-hf", device_map="sequential", torch_dtype=torch.float16
#         )
#         model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#         # greedy generation outputs
#         generated_ids = model.generate(
#             **model_inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False, dola_layers="low"
#         )
#         text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

#     @slow
#     @require_torch_accelerator
#     @require_read_token
#     def test_compile_static_cache(self):
#         # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
#         # work as intended. See https://github.com/pytorch/pytorch/issues/121943
#         if version.parse(torch.__version__) < version.parse("2.3.0"):
#             self.skipTest(reason="This test requires torch >= 2.3 to run.")

#         NUM_TOKENS_TO_GENERATE = 40
#         # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
#         # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.
#         EXPECTED_TEXT_COMPLETION = [
#             "Simply put, the theory of relativity states that 1) the speed of light is constant in all inertial "
#             "reference frames, and 2) the laws of physics are the same for all inertial reference frames.\nThe "
#             "theory of relativ",
#             "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, "
#             "my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
#         ]

#         prompts = [
#             "Simply put, the theory of relativity states that ",
#             "My favorite all time favorite condiment is ketchup.",
#         ]
#         tokenizer = MoshiAsrTokenizer.from_pretrained("meta-llama/MoshiAsr-2-7b-hf", pad_token="</s>", padding_side="right")
#         model = MoshiAsrForCausalLM.from_pretrained(
#             "meta-llama/MoshiAsr-2-7b-hf", device_map=torch_device, torch_dtype=torch.float16
#         )
#         inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

#         # Dynamic Cache
#         generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
#         dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

#         # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
#         generated_ids = model.generate(
#             **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
#         )
#         static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

#     @slow
#     @require_read_token
#     def test_export_static_cache(self):
#         if version.parse(torch.__version__) < version.parse("2.4.0"):
#             self.skipTest(reason="This test requires torch >= 2.4 to run.")

#         from transformers.integrations.executorch import (
#             TorchExportableModuleWithStaticCache,
#             convert_and_export_with_cache,
#         )

#         llama_models = {
#             "meta-llama/MoshiAsr-3.2-1B": [
#                 "Simply put, the theory of relativity states that 1) the speed of light is the same for all "
#                 "observers, regardless of their location, and 2) the laws of physics are the same for all observers"
#             ],
#         }

#         for llama_model_ckp, EXPECTED_TEXT_COMPLETION in llama_models.items():
#             # Load tokenizer
#             tokenizer = AutoTokenizer.from_pretrained(llama_model_ckp, pad_token="</s>", padding_side="right")
#             max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
#                 "input_ids"
#             ].shape[-1]

#             # Load model
#             device = "cpu"
#             dtype = torch.bfloat16
#             cache_implementation = "static"
#             attn_implementation = "sdpa"
#             batch_size = 1
#             model = MoshiAsrForCausalLM.from_pretrained(
#                 llama_model_ckp,
#                 device_map=device,
#                 torch_dtype=dtype,
#                 attn_implementation=attn_implementation,
#                 generation_config=GenerationConfig(
#                     use_cache=True,
#                     cache_implementation=cache_implementation,
#                     max_length=max_generation_length,
#                     cache_config={
#                         "batch_size": batch_size,
#                         "max_cache_len": max_generation_length,
#                         "device": device,
#                     },
#                 ),
#             )

#             prompts = ["Simply put, the theory of relativity states that "]
#             prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
#             prompt_token_ids = prompt_tokens["input_ids"]
#             max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

#             # Static Cache + export
#             exported_program = convert_and_export_with_cache(model)
#             ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
#                 exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
#             )
#             ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
#             self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
