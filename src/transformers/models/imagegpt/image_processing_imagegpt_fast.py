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
"""Fast Image processor class for ImageGPT."""
import PIL
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from ...image_processing_utils_fast import (
  BaseImageProcessorFast,
  DefaultFastImageProcessorKwargs,
  BatchFeature
)
from ...processing_utils import Unpack
from ...image_utils import (
    PILImageResampling,
    ImageInput,
    ChannelDimension, SizeDict
)
from ...utils import (
    auto_docstring,
    is_torch_available,
    TensorType
)

if is_torch_available():
    import torch

def squared_euclidean_distance_fast(a, b):
    b = b.T
    a2 = torch.sum(a ** 2, dim = 1)
    b2 = torch.sum(b ** 2, dim = 0)
    ab = torch.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d

def color_quantize_fast(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_fast(x, clusters)
    return np.argmin(d, axis=1)

class ImageGPTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_color_quantize: Optional[bool] = True
    clusters: Optional[np.ndarray] = None
    resample: Optional[PILImageResampling] = PILImageResampling.BILINEAR
    return_tensors: Optional[Union[str, TensorType]] = None,
    data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None

@auto_docstring
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    size = {"height": 256, "width": 256} # import get_size_dict?, can be overridden in preprocess
    do_resize = True
    do_normalize = True

    # Specific Kwargs
    do_color_quantize = True
    clusters = None
    resample = PILImageResampling.BILINEAR

    # not in base ##########
    image_mean = None # not in base, normalize uses a constant factor to divide pixel values
    image_std = None # not in base, normalize uses a constant factor to divide pixel values
    default_to_square = None # not in base
    crop_size = None # not in base
    do_center_crop = None # not in base
    do_rescale = None # not in base
    do_convert_rgb = None # not in base
    ############

    # initialize these arguments, pass it into super constructor
    def __init__(self, **kwargs: Unpack[ImageGPTFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    # _preprocessor has additional kwargs:
        # images, return_tensors, data_format, input_data_format

    # PUBLIC preprocess:
    def preprocess(self, images: ImageInput, **kwargs: Unpack[ImageGPTFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    # PRIVATE preprocess:
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        do_normalize: bool,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # TODO: Override
        # Resize to specific size
        # Normalize pixel values
        # Optionally color quantize into clusters
        # Return processed images in a specified tensor format
        pass

__all__ = ["ImageGPTImageProcessorFast"]
