<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# LayoutLMv3

LayoutLMv3 is a powerful multimodal transformer model designed specifically for Document AI tasks. What makes it unique is its unified approach to handling both text and images in documents, using a simple yet effective architecture that combines patch embeddings with transformer layers. Unlike its predecessor LayoutLMv2, it uses a more streamlined approach with patch embeddings (similar to ViT) instead of a CNN backbone.

The model is pre-trained on three key objectives:
1. Masked Language Modeling (MLM) for text understanding
2. Masked Image Modeling (MIM) for visual understanding
3. Word-Patch Alignment (WPA) for learning cross-modal relationships

This unified architecture and training approach makes LayoutLMv3 particularly effective for both text-centric tasks (like form understanding and receipt analysis) and image-centric tasks (like document classification and layout analysis).

[Paper](https://arxiv.org/abs/2204.08387) | [Official Checkpoints](https://huggingface.co/microsoft/layoutlmv3-base)

> [!TIP]
> Click on the right sidebar for more examples of how to use the model for different tasks!

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

# For document classification
classifier = pipeline("document-classification", model="microsoft/layoutlmv3-base")
result = classifier("document.jpg")

# For token classification (e.g., form understanding)
token_classifier = pipeline("token-classification", model="microsoft/layoutlmv3-base")
result = token_classifier("form.jpg")

# For question answering
qa = pipeline("document-question-answering", model="microsoft/layoutlmv3-base")
result = qa(question="What is the total amount?", image="receipt.jpg")
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor

# Load model and processor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

# Process inputs
image = Image.open("document.jpg").convert("RGB")
encoding = processor(image, return_tensors="pt")

# Get predictions
outputs = model(**encoding)
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli document-classification "document.jpg" --model microsoft/layoutlmv3-base
```

</hfoption>
</hfoptions>

For large models, you can use quantization to reduce memory usage:

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch

# Load model with 8-bit quantization
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    load_in_8bit=True,
    device_map="auto"
)

# Or with 4-bit quantization
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    load_in_4bit=True,
    device_map="auto"
)
```

## Notes

- In terms of data processing, LayoutLMv3 is identical to its predecessor [LayoutLMv2](layoutlmv2), except that:
    - images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format.
    - text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
  Due to these differences in data preprocessing, one can use [`LayoutLMv3Processor`] which internally combines a [`LayoutLMv3ImageProcessor`] (for the image modality) and a [`LayoutLMv3Tokenizer`]/[`LayoutLMv3TokenizerFast`] (for the text modality) to prepare all data for the model.
- Regarding usage of [`LayoutLMv3Processor`], we refer to the [usage guide](layoutlmv2#usage-layoutlmv2processor) of its predecessor.

## Model Details

### LayoutLMv3Config

[[autodoc]] LayoutLMv3Config

### LayoutLMv3FeatureExtractor

[[autodoc]] LayoutLMv3FeatureExtractor
    - __call__

### LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__

### LayoutLMv3ImageProcessor

[[autodoc]] LayoutLMv3ImageProcessor
    - preprocess

### LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__
    - save_vocabulary

### LayoutLMv3TokenizerFast

[[autodoc]] LayoutLMv3TokenizerFast
    - __call__

### LayoutLMv3Processor

[[autodoc]] LayoutLMv3Processor
    - __call__

### LayoutLMv3Model

[[autodoc]] LayoutLMv3Model

### LayoutLMv3ForSequenceClassification

[[autodoc]] LayoutLMv3ForSequenceClassification

### LayoutLMv3ForTokenClassification

[[autodoc]] LayoutLMv3ForTokenClassification

### LayoutLMv3ForQuestionAnswering

[[autodoc]] LayoutLMv3ForQuestionAnswering

### TFLayoutLMv3Model

[[autodoc]] TFLayoutLMv3Model

### TFLayoutLMv3ForSequenceClassification

[[autodoc]] TFLayoutLMv3ForSequenceClassification

### TFLayoutLMv3ForTokenClassification

[[autodoc]] TFLayoutLMv3ForTokenClassification

### TFLayoutLMv3ForQuestionAnswering

[[autodoc]] TFLayoutLMv3ForQuestionAnswering
