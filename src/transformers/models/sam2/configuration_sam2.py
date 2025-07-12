# coding=utf-8
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
"""SAM2 model configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig


logger = logging.get_logger(__name__)


class Sam2HieraDetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2HieraDetModel`]. It is used to instantiate
    a HieraDet model as defined in the original sam2 repo according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 96):
            The hidden dimension of the image encoder.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the image.
        image_size (`int`, *optional*, defaults to 1024):
            The size of the image.
        patch_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size of the patch.
        patch_stride (`int`, *optional*, defaults to 4):
            The stride of the patch.
        patch_padding (`int`, *optional*, defaults to 3):
            The padding of the patch.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The stochastic depth rate.
        q_pool (`int`, *optional*, defaults to 3):
            The number of q_pool stages.
        q_stride (`Tuple[int, int]`, *optional*, defaults to `[2, 2]`):
            The downsample stride between stages.
        stages (`Tuple[int, ...]`, *optional*, defaults to `[1, 2, 7, 2]`):
            The number of blocks per stage.
        dim_mul (`float`, *optional*, defaults to 2.0):
            The dimension multiplier factor at stage shift.
        head_mul (`float`, *optional*, defaults to 2.0):
            The head multiplier factor at stage shift.
        window_positional_embedding_background_size (`Tuple[int, int]`, *optional*, defaults to `[7, 7]`):
            The window size per stage when not using global attention.
        window_spec (`Tuple[int, ...]`, *optional*, defaults to `[8, 4, 14, 7]`):
            The window specifications for each stage.
        global_attention_blocks (`Tuple[int, ...]`, *optional*, defaults to `[5, 7, 9]`):
            The blocks where global attention is used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "vision_config"
    model_type = "sam2_hiera_det_model"

    def __init__(
        self,
        hidden_size=96,
        num_attention_heads=1,
        num_channels=3,
        image_size=1024,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=[2, 2],
        stages=[1, 2, 7, 2],
        dim_mul=2.0,
        head_mul=2.0,
        window_positional_embedding_background_size=[7, 7],
        window_spec=[8, 4, 14, 7],
        global_attention_blocks=[5, 7, 9],
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_positional_embedding_background_size = window_positional_embedding_background_size
        self.window_spec = window_spec
        self.global_attention_blocks = global_attention_blocks

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class Sam2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2VisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PretrainedConfig"]`, *optional*):
            Configuration for the vision backbone. This is used to instantiate the backbone using
            `AutoModel.from_config`.
        backbone_channel_list (`List[int]`, *optional*, defaults to `[768, 384, 192, 96]`):
            The list of channel dimensions for the backbone.
        backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[256, 256], [128, 128], [64, 64]]`):
            The spatial sizes of the feature maps from the backbone.
        fpn_hidden_size (`int`, *optional*, defaults to 256):
            The hidden dimension of the FPN.
        fpn_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size for the convolutions in the neck.
        fpn_stride (`int`, *optional*, defaults to 1):
            The stride for the convolutions in the neck.
        fpn_padding (`int`, *optional*, defaults to 0):
            The padding for the convolutions in the neck.
        fpn_top_down_levels (`List[int]`, *optional*, defaults to `[2, 3]`):
            The levels for the top-down FPN connections.
        fpn_interpolation_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation model for the FPN.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels from the FPN to use.
        fuse_type (`str`, *optional*, defaults to `"sum"`):
            The type of fusion to use in the neck.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "vision_config"
    model_type = "sam2_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    def __init__(
        self,
        backbone_config=None,
        backbone_channel_list=[768, 384, 192, 96],
        backbone_feature_sizes=[[256, 256], [128, 128], [64, 64]],
        fpn_hidden_size=256,
        fpn_kernel_size=1,
        fpn_stride=1,
        fpn_padding=0,
        fpn_top_down_levels=[2, 3],
        fpn_interpolation_mode="nearest",
        num_feature_levels=3,
        fuse_type="sum",
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = (
                backbone_config["model_type"] if "model_type" in backbone_config else "hiera"
            )
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif isinstance(backbone_config, (Sam2HieraDetConfig, TimmWrapperConfig)):
            backbone_config = backbone_config
        elif backbone_config is None:
            backbone_config = Sam2HieraDetConfig()

        self.backbone_config = backbone_config

        assert fuse_type in ["sum", "average"]
        # Neck
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_kernel_size = fpn_kernel_size
        self.fpn_stride = fpn_stride
        self.fpn_padding = fpn_padding
        self.fpn_top_down_levels = fpn_top_down_levels
        self.fpn_interpolation_mode = fpn_interpolation_mode
        self.fuse_type = fuse_type
        self.num_feature_levels = num_feature_levels

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class Sam2PromptEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2PromptEncoder`]. The [`Sam2PromptEncoder`]
    module is used to encode the input 2D points and bounding boxes.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        scale (`float`, *optional*, defaults to 1):
            The scale factor for the prompt encoder.
    """

    base_config_key = "prompt_encoder_config"

    def __init__(
        self,
        hidden_size=256,
        image_size=1024,
        patch_size=16,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale


class Sam2MaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MaskDecoder`]. It is used to instantiate a SAM 2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the SAM mask decoder.
        mlp_dim (`int`, *optional*, defaults to 2048):
            The dimension of the MLP in the two-way transformer.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            The number of hidden layers in the two-way transformer.
        num_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads in the two-way transformer.
        attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsample rate for the attention layers.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of multimask outputs.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The depth of the IoU head.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the IoU head.
        dynamic_multimask_via_stability (`bool`, *optional*, defaults to `True`):
            Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (`float`, *optional*, defaults to 0.05):
            The stability delta for the dynamic multimask.
        dynamic_multimask_stability_thresh (`float`, *optional*, defaults to 0.98):
            The stability threshold for the dynamic multimask.
        feed_forward_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the feed-forward network.
        two_way_transformer_activation (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the two-way transformer.

    """

    base_config_key = "mask_decoder_config"

    def __init__(
        self,
        hidden_size=256,
        hidden_act="gelu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        feed_forward_hidden_act="relu",
        two_way_transformer_activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.hidden_act = hidden_act
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.feed_forward_hidden_act = feed_forward_hidden_act
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # TwoWayTransformer configuration
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.two_way_transformer_activation = two_way_transformer_activation
        self.attention_downsample_rate = attention_downsample_rate


class Sam2MemoryAttentionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MemoryAttention`]. It is used to instantiate a SAM 2
    memory attention module according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        num_layers (`int`, *optional*, defaults to 4):
            The number of layers in the memory attention module.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the memory attention module.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            The dimension of the feedforward network in the memory attention module.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the memory attention module.
        rope_theta (`float`, *optional*, defaults to 10000):
            The Rope theta parameter.
        rope_feat_sizes (`Tuple[int, int]`, *optional*, defaults to `[64, 64]`):
            The feature sizes for the Rope positional encoding.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the memory attention.
        attention_downsample_rate (`int`, *optional*, defaults to 1):
            The downsample rate for the attention layers.
        rope_dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the Rope positional encoding.
        apply_pe_at_self_attn (`bool`, *optional*, defaults to `False`):
            Whether to apply positional encoding at the self-attention of the memory attention module.
        apply_pe_at_cross_attn_keys (`bool`, *optional*, defaults to `True`):
            Whether to apply positional encoding at the keys of the cross-attention of the memory attention module.
        apply_pe_at_cross_attn_queries (`bool`, *optional*, defaults to `False`):
            Whether to apply positional encoding at the queries of the cross-attention of the memory attention module.

    """

    base_config_key = "memory_attention_config"

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        hidden_act="relu",
        dim_feedforward=2048,
        dropout=0.1,
        rope_theta=10000,
        rope_feat_sizes=[64, 64],
        num_attention_heads=1,
        attention_downsample_rate=1,
        rope_dropout=0.1,
        apply_pe_at_self_attn=False,
        apply_pe_at_cross_attn_keys=True,
        apply_pe_at_cross_attn_queries=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_act = hidden_act
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.rope_dropout = rope_dropout
        self.apply_pe_at_self_attn = apply_pe_at_self_attn
        self.apply_pe_at_cross_attn_keys = apply_pe_at_cross_attn_keys
        self.apply_pe_at_cross_attn_queries = apply_pe_at_cross_attn_queries


class Sam2MemoryEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MemoryEncoder`]. It is used to instantiate a SAM 2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        output_channels (`int`, *optional*, defaults to 64):
            The number of output channels for the mask downsampler.
        mask_downsampler_embed_dim (`int`, *optional*, defaults to 256):
            The dimension of the mask downsampler embedding.
        mask_downsampler_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the mask downsampler.
        mask_downsampler_stride (`int`, *optional*, defaults to 2):
            The stride for the mask downsampler.
        mask_downsampler_padding (`int`, *optional*, defaults to 1):
            The padding for the mask downsampler.
        mask_downsampler_total_stride (`int`, *optional*, defaults to 16):
            The total stride for the mask downsampler.
        mask_downsampler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the mask downsampler.
        memory_fuser_num_layers (`int`, *optional*, defaults to 2):
            The number of layers in the memory fuser.
        memory_fuser_embed_dim (`int`, *optional*, defaults to 256):
            The dimension of the memory fuser embedding.
        memory_fuser_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size for the memory fuser.
        memory_fuser_padding (`int`, *optional*, defaults to 3):
            The padding for the memory fuser.
        memory_fuser_layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            The initial value for the layer scale in the memory fuser.
        memory_fuser_use_depthwise_conv (`bool`, *optional*, defaults to `True`):
            Whether to use a depthwise convolution for the memory fuser.
        memory_fuser_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the memory fuser.

    """

    base_config_key = "memory_encoder_config"

    def __init__(
        self,
        hidden_size=256,
        output_channels=64,
        mask_downsampler_embed_dim=256,
        mask_downsampler_kernel_size=3,
        mask_downsampler_stride=2,
        mask_downsampler_padding=1,
        mask_downsampler_total_stride=16,
        mask_downsampler_hidden_act="gelu",
        memory_fuser_num_layers=2,
        memory_fuser_embed_dim=256,
        memory_fuser_kernel_size=7,
        memory_fuser_padding=3,
        memory_fuser_layer_scale_init_value=1e-6,
        memory_fuser_use_depthwise_conv=True,
        memory_fuser_hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            mask_downsampler_stride
            ** int(math.log2(mask_downsampler_total_stride) // math.log2(mask_downsampler_stride))
            == mask_downsampler_total_stride
        )

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.mask_downsampler_embed_dim = mask_downsampler_embed_dim
        self.mask_downsampler_kernel_size = mask_downsampler_kernel_size
        self.mask_downsampler_stride = mask_downsampler_stride
        self.mask_downsampler_padding = mask_downsampler_padding
        self.mask_downsampler_total_stride = mask_downsampler_total_stride
        self.mask_downsampler_hidden_act = mask_downsampler_hidden_act
        self.memory_fuser_num_layers = memory_fuser_num_layers
        self.memory_fuser_embed_dim = memory_fuser_embed_dim
        self.memory_fuser_kernel_size = memory_fuser_kernel_size
        self.memory_fuser_padding = memory_fuser_padding
        self.memory_fuser_layer_scale_init_value = memory_fuser_layer_scale_init_value
        self.memory_fuser_use_depthwise_conv = memory_fuser_use_depthwise_conv
        self.memory_fuser_hidden_act = memory_fuser_hidden_act


class Sam2Config(PretrainedConfig):
    r"""
    [`Sam2Config`] is the configuration class to store the configuration of a [`Sam2Model`]. It is used to instantiate a
    SAM2 model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2 Hiera-B+
    [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `Sam2VisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2VisionConfig`].
        prompt_encoder_config (Union[`dict`, `Sam2PromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2PromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `Sam2MaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MaskDecoderConfig`].
        memory_attention_config (Union[`dict`, `Sam2MemoryAttentionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MemoryAttentionConfig`].
        memory_encoder_config (Union[`dict`, `Sam2MemoryEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MemoryEncoderConfig`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for parameter initialization.
        num_maskmem (`int`, *optional*, defaults to 7):
            The number of memory slots for the mask memory.
        image_size (`int`, *optional*, defaults to 1024):
            The size of the input images.
        sigmoid_scale_for_mem_enc (`float`, *optional*, defaults to 20.0):
            Scale factor for the sigmoid function in the memory encoder.
        sigmoid_bias_for_mem_enc (`float`, *optional*, defaults to -10.0):
            Bias for the sigmoid function in the memory encoder.
        binarize_mask_from_pts_for_mem_enc (`bool`, *optional*, defaults to `True`):
            Whether to binarize the mask from points for the memory encoder.
        enable_occlusion_spatial_embedding (`bool`, *optional*, defaults to `True`):
            Whether to enable spatial embedding for occlusions.
        multimask_output_in_sam (`bool`, *optional*, defaults to `True`):
            Whether to output multiple masks from the SAM head.
        multimask_min_pt_num (`int`, *optional*, defaults to 0):
            The minimum number of points to trigger multimask output.
        multimask_max_pt_num (`int`, *optional*, defaults to 1):
            The maximum number of points to trigger multimask output.
        multimask_output_for_tracking (`bool`, *optional*, defaults to `True`):
            Whether to use multimask output for tracking.
        non_overlap_masks_for_mem_enc (`bool`, *optional*, defaults to `False`):
            Whether to enforce non-overlapping masks for the memory encoder.
        max_object_pointers_in_encoder (`int`, *optional*, defaults to 16):
            The maximum number of object pointers in the encoder.
        enable_temporal_pos_encoding_for_object_pointers (`bool`, *optional*, defaults to `True`):
            Whether to enable temporal positional encoding for object pointers.
        project_temporal_pos_encoding_in_object_pointers (`bool`, *optional*, defaults to `True`):
            Whether to project temporal positional encoding in object pointers.
        preserve_temporal_direction_in_object_pointers (`bool`, *optional*, defaults to `True`):
            Whether to preserve temporal direction in object pointers.
        fill_hole_area (`int`, *optional*, defaults to 8):
            The maximum area of holes to fill in the masks.
        non_overlap_masks (`bool`, *optional*, defaults to `False`):
            Whether to enforce non-overlapping masks.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam2VisionConfig,
    ...     Sam2PromptEncoderConfig,
    ...     Sam2MaskDecoderConfig,
    ...     Sam2MemoryAttentionConfig,
    ...     Sam2MemoryEncoderConfig,
    ...     Sam2Model,
    ... )

    >>> # Initializing a Sam2Config with `"facebook/hiera-base-plus"` style configuration
    >>> configuration = Sam2config()

    >>> # Initializing a Sam2Model (with random weights) from the `"facebook/sam-vit-huge"` style configuration
    >>> model = Sam2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam2Config from a Sam2VisionConfig, Sam2MemoryAttentionConfig, and Sam2MemoryEncoderConfig

    >>> # Initializing SAM2 vision encoder, memory attention, and memory encoder configurations
    >>> vision_config = Sam2VisionConfig()
    >>> prompt_encoder_config = Sam2PromptEncoderConfig()
    >>> mask_decoder_config = Sam2MaskDecoderConfig()
    >>> memory_attention_config = Sam2MemoryAttentionConfig()
    >>> memory_encoder_config = Sam2MemoryEncoderConfig()

    >>> config = Sam2Config(vision_config, prompt_encoder_config, mask_decoder_config, memory_attention_config, memory_encoder_config)
    ```"""

    model_type = "sam2"
    sub_configs = {
        "vision_config": Sam2VisionConfig,
        "prompt_encoder_config": Sam2PromptEncoderConfig,
        "mask_decoder_config": Sam2MaskDecoderConfig,
        "memory_attention_config": Sam2MemoryAttentionConfig,
        "memory_encoder_config": Sam2MemoryEncoderConfig,
    }

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        memory_attention_config=None,
        memory_encoder_config=None,
        initializer_range=0.02,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        binarize_mask_from_pts_for_mem_enc=True,
        enable_occlusion_spatial_embedding=True,
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        non_overlap_masks_for_mem_enc=False,
        max_object_pointers_in_encoder=16,
        enable_temporal_pos_encoding_for_object_pointers=True,
        project_temporal_pos_encoding_in_object_pointers=True,
        preserve_temporal_direction_in_object_pointers=True,
        fill_hole_area=8,
        non_overlap_masks=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vision_config = vision_config if vision_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}
        memory_attention_config = memory_attention_config if memory_attention_config is not None else {}
        memory_encoder_config = memory_encoder_config if memory_encoder_config is not None else {}

        if isinstance(vision_config, Sam2VisionConfig):
            vision_config = vision_config.to_dict()
        if isinstance(prompt_encoder_config, Sam2PromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam2MaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()
        if isinstance(memory_attention_config, Sam2MemoryAttentionConfig):
            memory_attention_config = memory_attention_config.to_dict()
        if isinstance(memory_encoder_config, Sam2MemoryEncoderConfig):
            memory_encoder_config = memory_encoder_config.to_dict()

        self.vision_config = Sam2VisionConfig(**vision_config)
        self.prompt_encoder_config = Sam2PromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam2MaskDecoderConfig(**mask_decoder_config)
        self.memory_attention_config = Sam2MemoryAttentionConfig(**memory_attention_config)
        self.memory_encoder_config = Sam2MemoryEncoderConfig(**memory_encoder_config)

        self.initializer_range = initializer_range
        self.num_maskmem = num_maskmem  # default 1 input frame + 6 previous frames
        self.image_size = image_size
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc  # scale factor for mask sigmoid prob
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc  # bias factor for mask sigmoid prob
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.enable_occlusion_spatial_embedding = enable_occlusion_spatial_embedding
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.max_object_pointers_in_encoder = max_object_pointers_in_encoder
        self.enable_temporal_pos_encoding_for_object_pointers = enable_temporal_pos_encoding_for_object_pointers
        self.project_temporal_pos_encoding_in_object_pointers = project_temporal_pos_encoding_in_object_pointers
        self.preserve_temporal_direction_in_object_pointers = preserve_temporal_direction_in_object_pointers

        # post-processing parameters
        self.fill_hole_area = fill_hole_area  # area threshold for filling holes in masks
        self.non_overlap_masks = non_overlap_masks  # whether to apply non-overlapping constraints on output masks


__all__ = [
    "Sam2Config",
    "Sam2HieraDetConfig",
    "Sam2VisionConfig",
    "Sam2PromptEncoderConfig",
    "Sam2MaskDecoderConfig",
    "Sam2MemoryAttentionConfig",
    "Sam2MemoryEncoderConfig",
]
