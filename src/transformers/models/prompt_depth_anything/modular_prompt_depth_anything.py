from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.models.depth_anything.configuration_depth_anything import DepthAnythingConfig
from transformers.models.depth_anything.modeling_depth_anything import (
    DepthAnythingDepthEstimationHead,
    DepthAnythingFeatureFusionLayer,
    DepthAnythingForDepthEstimation,
    DepthAnythingNeck,
    DepthAnythingReassembleLayer,
    DepthAnythingReassembleStage,
)
from transformers.utils.generic import torch_int

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel


_CONFIG_FOR_DOC = "PromptDepthAnythingConfig"

PROMPT_DEPTH_ANYTHING_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PromptDepthAnythingConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PROMPT_DEPTH_ANYTHING_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        prompt_depth (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            Prompt depth.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class PromptDepthAnythingConfig(DepthAnythingConfig):
    model_type = "prompt_depth_anything"


class PromptDepthAnythingResidualLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convolution1 = nn.Conv2d(
            1,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.activation1 = nn.ReLU(False)

        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.activation2 = nn.ReLU(False)

        self.convolution3 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, prompt_depth: torch.Tensor) -> torch.Tensor:
        residual = prompt_depth
        residual = self.convolution1(residual)
        residual = self.activation1(residual)
        residual = self.convolution2(residual)
        residual = self.activation2(residual)
        residual = self.convolution3(residual)
        return residual


class PromptDepthAnythingFeatureFusionLayer(DepthAnythingFeatureFusionLayer):
    def __init__(self, config):
        super().__init__(config)
        self.residual_layer_depth = PromptDepthAnythingResidualLayer(config)

    def forward(self, hidden_state, residual=None, size=None, prompt_depth=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)

        if prompt_depth is not None:
            prompt_depth = nn.functional.interpolate(
                prompt_depth, hidden_state.shape[2:], mode="bilinear", align_corners=False
            )
            res = self.residual_layer_depth(prompt_depth)
            hidden_state = hidden_state + res

        modifier = {"scale_factor": 2} if size is None else {"size": size}

        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=True,
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class PromptDepthAnythingFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(PromptDepthAnythingFeatureFusionLayer(config))

    def forward(self, hidden_states, size=None, prompt_depth=None):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        fused_hidden_state = None

        for idx, (hidden_state, layer) in enumerate(zip(hidden_states, self.layers)):
            size = hidden_states[idx + 1].shape[2:] if idx != (len(hidden_states) - 1) else None

            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = layer(hidden_state, size=size, prompt_depth=prompt_depth)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size, prompt_depth=prompt_depth)

            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class PromptDepthAnythingDepthEstimationHead(DepthAnythingDepthEstimationHead):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height, patch_width) -> torch.Tensor:
        hidden_states = hidden_states[self.head_in_index]

        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (torch_int(patch_height * self.patch_size), torch_int(patch_width * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth) * self.max_depth
        predicted_depth = predicted_depth.squeeze(dim=1)  # shape (batch_size, height, width)

        return predicted_depth


# Copied from transformers.models.dpt.modeling_dpt.DPTPreTrainedModel with DPT->PromptDepthAnything,dpt->prompt_depth_anything
class PromptDepthAnythingPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PromptDepthAnythingConfig
    base_model_prefix = "prompt_depth_anything"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class PromptDepthAnythingReassembleLayer(DepthAnythingReassembleLayer):
    def __init__(self, config, channels, factor):
        super().__init__(config, channels, factor)
        self.projection = nn.Conv2d(in_channels=config.reassemble_hidden_size, out_channels=channels, kernel_size=1)

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=torch_int(1 / factor), padding=1)


class PromptDepthAnythingReassembleStage(DepthAnythingReassembleStage):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Take the patch embeddings and reshape them to image-like feature representations.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList()
        for channels, factor in zip(config.neck_hidden_sizes, config.reassemble_factors):
            self.layers.append(PromptDepthAnythingReassembleLayer(config, channels=channels, factor=factor))


class PromptDepthAnythingNeck(DepthAnythingNeck):
    def __init__(self, config):
        super().__init__(config)
        self.reassemble_stage = PromptDepthAnythingReassembleStage(config)
        self.fusion_stage = PromptDepthAnythingFeatureFusionStage(config)

    def forward(
        self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None, prompt_depth=None
    ) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features, prompt_depth=prompt_depth)

        return output


@add_start_docstrings(
    """
    Prompt Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """,
    PROMPT_DEPTH_ANYTHING_START_DOCSTRING,
)
class PromptDepthAnythingForDepthEstimation(DepthAnythingForDepthEstimation):
    def __init__(self, config):
        super().__init__(config)
        self.neck = PromptDepthAnythingNeck(config)
        self.head = PromptDepthAnythingDepthEstimationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(PROMPT_DEPTH_ANYTHING_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        prompt_depth: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("depth-anything/promptda_vits_hf")
        >>> model = AutoModelForDepthEstimation.from_pretrained("depth-anything/promptda_vits_hf")

        >>> prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
        >>> prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)
        >>> prompt_depth = torch.tensor((np.asarray(prompt_depth) / 1000.0).astype(np.float32))
        >>> prompt_depth = prompt_depth.unsqueeze(0).unsqueeze(0)

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values=inputs.pixel_values, prompt_depth=prompt_depth)

        >>> # interpolate to original size
        >>> post_processed_output = image_processor.post_process_depth_estimation(
        ...     outputs,
        ...     target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 1000.
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint16")) # mm
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        if prompt_depth is not None:
            # normalize prompt depth
            B = prompt_depth.shape[0]
            depth_min, depth_max = (
                torch.min(prompt_depth.reshape(B, -1), dim=1).values,
                torch.max(prompt_depth.reshape(B, -1), dim=1).values,
            )
            invalid_mask = (depth_max - depth_min) <= 0
            if invalid_mask.any():
                depth_max[invalid_mask] = depth_min[invalid_mask] + 1e-6
            depth_min, depth_max = depth_min.view(B, 1, 1, 1), depth_max.view(B, 1, 1, 1)
            prompt_depth = (prompt_depth - depth_min) / (depth_max - depth_min)
            # normalize done

        hidden_states = self.neck(hidden_states, patch_height, patch_width, prompt_depth=prompt_depth)

        predicted_depth = self.head(hidden_states, patch_height, patch_width)
        if prompt_depth is not None:
            # denormalize predicted depth
            depth_min, depth_max = depth_min.squeeze(1), depth_max.squeeze(1)
            predicted_depth = predicted_depth * (depth_max - depth_min) + depth_min
            # denormalize done

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


__all__ = [
    "PromptDepthAnythingConfig",
    "PromptDepthAnythingForDepthEstimation",
    "PromptDepthAnythingPreTrainedModel",
]
