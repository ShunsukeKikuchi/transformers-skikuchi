<!--Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# EoMT

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Encoder-only Mask Transformer (EoMT) model was introduced in the CVPR 2025 Highlight Paper [Your ViT is Secretly an Image Segmentation Model](https://www.tue-mps.org/eomt) by Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus.
EoMT reveals Vision Transformers can perform image segmentation efficiently without task-specific components.

The abstract from the paper is the following:

*Vision Transformers (ViTs) have shown remarkable performance and scalability across various computer vision tasks. To apply single-scale ViTs to image segmentation, existing methods adopt a convolutional adapter to generate multi-scale features, a pixel decoder to fuse these features, and a Transformer decoder that uses the fused features to make predictions. In this paper, we show that the inductive biases introduced by these task-specific components can instead be learned by the ViT itself, given sufficiently large models and extensive pre-training. Based on these findings, we introduce the Encoder-only Mask Transformer (EoMT), which repurposes the plain ViT architecture to conduct image segmentation. With large-scale models and pre-training, EoMT obtains a segmentation accuracy similar to state-of-the-art models that use task-specific components. At the same time, EoMT is significantly faster than these methods due to its architectural simplicity, e.g., up to 4x faster with ViT-L. Across a range of model sizes, EoMT demonstrates an optimal balance between segmentation accuracy and prediction speed, suggesting that compute resources are better spent on scaling the ViT itself rather than adding architectural complexity.*

This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali).
The original code can be found [here](https://github.com/tue-mps/eomt).

## Usage Examples

- Use the Hugging Face implementation of EoMT for inference with pre-trained models.

### Semantic Segmentation

```python
from transformers import EoMTImageProcessor, EoMTForUniversalSegmentation
import numpy as np
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt


model_id = ""
processor = EoMTImageProcessor.from_pretrained(model_id)
model = EoMTForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(
    requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
)

# Preprocess the image for semantic segmentation by setting `segmentation_type` arg
inputs = processor(
    images=image,
    segmentation_type="semantic",
    return_tensors="pt",
)

# Remove crop offsets from inputs — used later for post-processing.
crops_offset = inputs.pop("crops_offset")

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
original_image_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_semantic_segmentation(
    outputs,
    crops_offset=crops_offset,
    original_image_sizes=original_image_sizes,
)

# Visualize the segmentation mask
plt.imshow(preds[0])
plt.axis("off")
plt.title("Semantic Segmentation")
plt.show()
```

### Instance Segmentation

```python
from transformers import EoMTImageProcessor, EoMTForUniversalSegmentation
import numpy as np
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt


model_id = ""
processor = EoMTImageProcessor.from_pretrained(model_id)
model = EoMTForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(
    requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
)

# Preprocess the image for semantic segmentation by setting `segmentation_type` arg
inputs = processor(
    images=image,
    segmentation_type="instance",
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
original_image_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_semantic_segmentation(
    outputs,
    original_image_sizes=original_image_sizes,
    stuff_classes = [0,45] # Pass the list of stuff classes to exclude from mask.
)

# Visualize the segmentation mask
plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Instance Segmentation")
plt.show()
```

### Panoptic Segmentation

```python
from transformers import EoMTImageProcessor, EoMTForUniversalSegmentation
import numpy as np
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt


model_id = ""
processor = EoMTImageProcessor.from_pretrained(model_id)
model = EoMTForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(
    requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
)

# Preprocess the image for panoptic segmentation by setting `segmentation_type` arg
inputs = processor(
    images=image,
    segmentation_type="panoptic",
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
original_image_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_panoptic_segmentation(
    outputs,
    original_image_sizes=original_image_sizes,
)

# Visualize the panoptic segmentation mask
plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Panoptic Segmentation")
plt.show()
```

## EoMTImageProcessor

[[autodoc]] EoMTImageProcessor

## EoMTConfig

[[autodoc]] EoMTConfig

## EoMTForUniversalSegmentation

[[autodoc]] EoMTForUniversalSegmentation
    - forward