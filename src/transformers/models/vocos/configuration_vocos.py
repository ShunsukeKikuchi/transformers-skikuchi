# coding=utf-8
# Copyright 2024 Descript and The HuggingFace Inc. team. All rights reserved.
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
"""Vocos model configuration"""

from typing import Optional, Sequence

from transformers import EncodecConfig

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class VocosConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VocosModel`]. It is used to
    instantiate a Vocos vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Manel/Vocos](https://huggingface.co/Manel/Vocos) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        input_channels (`int`, *optional*, defaults to 100):
            Number of mel‐spectrogram input channels (i.e. number of mel filter bins).
        hidden_dim (`int`, *optional*, defaults to 512):
            Hidden dimension for the ConvNeXt backbone.
        intermediate_dim (`int`, *optional*, defaults to 1536):
            Dimension of the feed‐forward layers inside each ConvNeXt block.
        num_layers (`int`, *optional*, defaults to 8):
            Number of ConvNeXt blocks to stack.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for depthwise convolutions.
        padding (`int`, *optional*, defaults to 3):
            Padding applied to those convolutions.
        layer_scale_init_value (`float`, *optional*, defaults to `1/8`):
            Initial value for layer‐scale (if >0, enables per‐block scaling).
        use_adaptive_norm (`bool`, *optional*, defaults to `False`):
            Whether to use adaptive layer normalization .
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for all LayerNorm operations.
        n_fft (`int`, *optional*, defaults to 1024):
            FFT size for STFT/ISTFT used in VocosISTFT head.
        hop_length (`int`, *optional*, defaults to 256):
            Hop length between STFT frames used in VocosISTFT head.
        spec_padding (`str`, *optional*, defaults to `"center"`):
            Padding mode for spectrogram inversion (`"center"` or `"same"`).

    Example:

    ```python
    >>> from transformers import VocosModel, VocosConfig
    >>> config = VocosConfig()
    >>> model = VocosModel(config)
    ```
    """

    model_type = "vocos"

    def __init__(
        self,
        input_channels: int = 100,
        hidden_dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        kernel_size: int = 7,
        padding: int = 3,
        layer_scale_init_value: float = 1 / 8,
        use_adaptive_norm: bool = False,
        layer_norm_eps: float = 1e-6,
        n_fft: int = 1024,
        hop_length: int = 256,
        spec_padding: str = "center",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer_scale_init_value = layer_scale_init_value
        self.use_adaptive_norm = use_adaptive_norm
        self.layer_norm_eps = layer_norm_eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec_padding = spec_padding


class VocosWithEncodecConfig(VocosConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`VocosWithEncodecModel`]. It extends [`VocosConfig`] by adding EnCodec‐specific parameters
    for audio encoding and quantization. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Manel/Vocos-Encodec](https://huggingface.co/Manel/Vocos-Encodec) architecture.

    Args:
        encodec_config (Union[Dict, EncodecConfig], *optional*):
            Configuration for the neural codec model EnCodec.
        train_codebooks (`bool`, *optional*, defaults to `False`):
            Whether to finetune the EnCodec codebook embeddings.
        bandwidths (`Sequence[float]`, *optional*, defaults to `(1.5, 3.0, 6.0, 12.0)`):
            Supported target bandwidths in kbps, This determines
            the number of quantizers/codebooksused in RVQ part
            of Encodec [2, 4, 6, 8].
        input_channels (`int`, *optional*, defaults to 128):
            Number of mel‐spectrogram input channels.
        hidden_dim (`int`, *optional*, defaults to 384):
            Hidden dimension for the ConvNeXt backbone.
        intermediate_dim (`int`, *optional*, defaults to 1152):
            Dimension of feed‐forward layers in ConvNeXt.
        num_layers (`int`, *optional*, defaults to 8):
            Number of ConvNeXt blocks.
        use_adaptive_norm (`bool`, *optional*, defaults to `True`):
            Whether to use adaptive layer normalization.
        adanorm_num_embeddings (`int`, *optional*, defaults to 4):
            Number of the embeddings in adaptive Norm layer.
        n_fft (`int`, *optional*, defaults to 1280):
            FFT size for STFT/ISTFT used in VocosISTFT head.
        hop_length (`int`, *optional*, defaults to 320):
            Hop length for STFT/ISTFT used in VocosISTFT head.
        spec_padding (`str`, *optional*, defaults to `"same"`):
            Padding mode for spectrogram inversion (`"center"` or `"same"`).

    Example:

    ```python
    >>> from transformers import VocosWithEncodecModel, VocosWithEncodecConfig
    >>> config = VocosWithEncodecConfig()
    >>> model = VocosWithEncodecModel(config)
    ```
    """

    model_type = "vocos_with_encodec"
    sub_configs = {"encodec_config": EncodecConfig}

    def __init__(
        self,
        encodec_config: Optional[dict] = None,
        train_codebooks: bool = False,
        bandwidths: Sequence[float] = (1.5, 3.0, 6.0, 12.0),
        input_channels: int = 128,
        hidden_dim: int = 384,
        intermediate_dim: int = 1152,
        num_layers: int = 8,
        use_adaptive_norm: bool = True,
        adanorm_num_embeddings: int = 4,
        n_fft: int = 1280,
        hop_length: int = 320,
        spec_padding: str = "same",
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            use_adaptive_norm=use_adaptive_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            spec_padding=spec_padding,
            **kwargs,
        )

        if encodec_config is None:
            self.encodec_config = EncodecConfig()
        elif isinstance(encodec_config, dict):
            self.encodec_config = EncodecConfig(**encodec_config)
        elif isinstance(encodec_config, EncodecConfig):
            self.encodec_config = encodec_config

        self.train_codebooks = train_codebooks
        self.bandwidths = list(bandwidths)
        self.adanorm_num_embeddings = adanorm_num_embeddings


__all__ = ["VocosConfig", "VocosWithEncodecConfig"]
