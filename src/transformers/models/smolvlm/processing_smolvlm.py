# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for SmolVLM.
"""

import re
from itertools import accumulate
from datetime import timedelta
import decord
from PIL import Image
from decord import VideoReader
decord.bridge.set_bridge("torch")
from num2words import num2words

from typing import TYPE_CHECKING, Dict, List, Optional, Union, Any, Tuple

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, BatchEncoding, TextInput
from ...utils import logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
# DEFAULT_VIDEO_INTRO = "Here are some frames sampled from a video:"
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames "
    "from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_IMAGE_INTRO = "Here are some images: "
DEFAULT_MEDIA_OUTTRO = "Now answer the following question: "
FRAME_TIMESTAMP_MESSAGE = "Frame from {timestamp}:"


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")
    
def is_str(val) -> bool:
    return isinstance(val, str)

def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def load_video(
    path: str,
    max_frames: int = 64,
    target_fps: float = 1.0,
    skip_secs: float = 1.0
) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads a video from `path` using decord, sampling up to `max_frames` frames.
    After deduplicating indices (e.g., to handle rounding collisions), each frame
    is decoded into a PIL Image (in RGB mode). Timestamps are generated in "MM:SS" format
    based on the frame index over `native_fps`.

    Args:
        path (str): Path to the video file (e.g., MP4).
        max_frames (int): Hard cap on how many frames we ever pick in total.
        target_fps (float): Target approximate sampling rate in frames per second.
        skip_secs (float): Number of seconds to skip at the beginning and end if 
            the video is sufficiently long ((duration - 2*skip_secs) > max_frames * target_fps).
    
    Returns:
        Tuple[List[Image.Image], List[str]]:
          - A list of PIL.Image objects corresponding to each selected frame.
          - A list of parallel timestamps ("MM:SS" strings), one per selected frame.
    """
    try:
        # Use decord with single-thread and CPU context
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    except Exception as e:
        raise RuntimeError(f"Failed to open video '{path}': {e}")

    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"Video '{path}' has 0 frames.")

    # Fallback to 30 if native_fps is None or zero
    native_fps = vr.get_avg_fps() or 30.0
    duration_seconds = total_frames / native_fps

    # Estimate how many frames we'd get if we sample at `target_fps`.
    estimated_frames = int(round(target_fps * duration_seconds)) if target_fps > 0 else max_frames
    desired_frames = min(estimated_frames, max_frames)
    if desired_frames < 1:
        desired_frames = 1

    start_idx = 0
    end_idx = total_frames - 1

    # Centered skip if we want fewer frames than max_frames
    if desired_frames < max_frames:
        leftover = total_frames - desired_frames
        start_idx = leftover // 2
        end_idx = total_frames - (leftover - start_idx)
    # Otherwise, if video is long enough, skip a bit from start and end
    elif skip_secs > 0 and (duration_seconds - 2 * skip_secs) > (max_frames * target_fps):
        start_idx = int(skip_secs * native_fps)
        end_idx = int(total_frames - skip_secs * native_fps)

    # Ensure valid start / end
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, total_frames - 1)
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = total_frames - 1

    # Uniformly sample the desired number of frames from [start_idx..end_idx]
    frames_idx = np.linspace(start_idx, end_idx, desired_frames, dtype=int)
    frames_idx = np.unique(frames_idx).tolist()

    # Read frames from decord
    try:
        frames_tensor = vr.get_batch(frames_idx).cpu().numpy()  # (N, H, W, C)
    except Exception as e:
        raise RuntimeError(f"Failed to read frames from '{path}': {e}")

    # Convert to PIL Images
    frames_out = [Image.fromarray(arr).convert("RGB") for arr in frames_tensor]

    # Build timestamps (MM:SS) for each selected frame index
    timestamps = []
    for idx in frames_idx:
        sec = idx / native_fps
        mm = int(sec // 60)
        ss = int(sec % 60)
        timestamps.append(f"{mm:02d}:{ss:02d}")

    return frames_out, timestamps, duration_seconds
    

def load_video_from_disk_or_url(path_or_url: str, sampling_fps: int = 1, max_frames: int = 48):
    """
    Minimal example of loading a video or frames from a URL/local path.
    Returns: (frames, timestamps, duration_seconds).
    This can be replaced by a more robust version with decord or ffmpeg, etc.
    """
    if is_url(path_or_url):
        ## load video
        frames = 0
    else:
        frames, timestamps, duration_secondsload_video = load_video(path_or_url, max_frames, sampling_fps)
        
    return frames, timestamps, duration_seconds

def _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}" + f"<row_{n_h + 1}_col_{n_w + 1}>" + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_img_token
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token
    )


class SmolVLMImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: Optional[bool]
    max_image_size: Optional[Dict[str, int]]


class SmolVLMProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SmolVLMImagesKwargs

    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
        },
    }


SmolVLMProcessorKwargs.__annotations__["images_kwargs"] = SmolVLMImagesKwargs  # python 3.8 compatibility


class SmolVLMProcessor(ProcessorMixin):
    r"""
    Constructs a SmolVLM processor which wraps a LLama tokenizer and SmolVLM image processor into a single processor.

    [`SmolVLMProcessor`] offers all the functionalities of [`SmolVLMImageProcessor`] and [`SmolVLMTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`SmolVLMImageProcessor`):
            An instance of [`SmolVLMImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["image_seq_len", "chat_template"]
    image_processor_class = "SmolVLMImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 169, chat_template: str = None, sampling_fps = 1, video_frame_size=384, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
            
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True)
        self.image_token = AddedToken("<image>", normalized=False, special=True)
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)
        self.global_image_tag = "<global-img>"  # https://github.com/huggingface/transformers/pull/32473/files/8063e5e17362571b693f1db95167f5443a3be1b2#r1734825341
        self.image_seq_len = image_seq_len
        self.sampling_fps = sampling_fps
        self.video_frame_size = video_frame_size

        # This regex matches one or more occurrences of <global-img> tags (optionally surrounded by newline characters)
        # or <row_x_col_y> tags (where x and y are digits, also optionally surrounded by newline characters).
        self._regex_to_remove_extra_special_tokens = re.compile(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+")

        tokens_to_add = {
            "additional_special_tokens": [
                self.fake_image_token,
                self.image_token,
                self.end_of_utterance_token,
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _extract_images_from_prompts(self, prompts):
        prompt_images = []
        for prompt in prompts:
            images = []
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))
            prompt_images.append(images)
        return prompt_images

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        video: Union[str, List[ImageInput], List[List[ImageInput]]] = None,
        video_frame_sampling_fps: int = 1, 
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        audio=None,
        image_seq_len: Optional[int] = None,
        **kwargs: Unpack[SmolVLMProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        Example:

        ```python
        >>> import requests
        >>> from transformers import SmolVLMProcessor
        >>> from transformers.image_utils import load_image

        >>> processor = SmolVLMProcessor.from_pretrained("HuggingFaceM4/SmolVLM-8B-Llama3")
        >>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [[image1], [image2]]

        >>> text = [
        ...     "<image>In this image, we see",
        ...     "bla bla bla<image>",
        ... ]
        >>> outputs = processor(images=images, text=text, return_tensors="pt", padding=True)
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        ['<|begin_of_text|><fake_token_around_image><global-img>((<image>)*169)<fake_token_around_image> In this image, we see', '<|reserved_special_token_0|><|reserved_special_token_0|><|reserved_special_token_0|><|begin_of_text|>bla bla bla<fake_token_around_image><global-img>((<image>)*169)<fake_token_around_image>']
        ```

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `List[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            image_seq_len (`int`, *optional*):
                The length of the image sequence. If not provided, the default value of self.image_seq_len is used.
                image_seq_len should be equal to int(((image_size // patch_size) ** 2) / (scale_factor**2))
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is not None and messages is not None:
            raise ValueError("You must provide only one of `text` or `messages'.")
            
        if text is None and images is None and video is None:
            raise ValueError("You must provide one of `text`, `images` or `video'.")

        if images is not None and video is not None:
            raise ValueError("You must provide either `images` or `video', not both.")

        output_kwargs = self._merge_kwargs(
            SmolVLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        n_images_in_text = []
        n_images_in_images = []
        inputs = BatchFeature()

        if images is not None:
            if messages is not None and text is None:
                text = self.apply_chat_template(messages)
                
            self.process_images(inputs, text, images)
            
        if video is not None:
            ## TODO, fix this
            if is_str(video) or is_url(video):
                # Single path/URL
                frames, timestamps, duration_sec = load_video_from_disk_or_url(
                    video, sampling_fps=video_frame_sampling_fps
                )
                images = [frames]
                
            elif isinstance(video, (list, tuple)):
                if video and is_image_or_image_url(video[0]):
                    # => single list of frames => wrap as [video]
                    frames = list(video)
                    images = [frames]
                    if messages is not None and text is None:
                        # Build naive timestamps
                        timestamps = []
                        for i in range(len(frames)):
                            mm = int(i // (60 * video_frame_sampling_fps))
                            ss = int(i % (60 * video_frame_sampling_fps))
                            ts_str = f"{mm:02d}:{ss:02d}"
                            timestamps.append(ts_str)
                        duration_sec = max(len(frames) - 1, 0) / float(video_frame_sampling_fps)
                else:
                    raise ValueError("Invalid format for `video` argument when it's a list/tuple.")
            
            if messages is not None and text is None:
                text = self.apply_chat_template(
                    messages, frames=len(frames), timestamps=timestamps, duration_sec=duration_sec
                )
            else:
                raise ValueError("Invalid `video` format. Must be string/URL, list of frames, or nested frames.")
                
            self.process_images(inputs, text, images, output_kwargs, do_image_splitting=False, image_processor_size=self.video_frame_size)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token.content} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        return inputs

    def process_images(self, inputs, text, images, output_kwargs, do_image_splitting=None, image_processor_size=None):
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token.content) for sample in text]
            
        if is_image_or_image_url(images):
            images = [[images]]
        elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
            if text is not None:
                if sum(n_images_in_text) != len(images):
                    raise ValueError(
                        f"The total number of {self.image_token.content} tokens in the prompts should be the same as the number of images passed."
                        f" Found {sum(n_images_in_text)} {self.image_token.content} tokens and {len(images)} images."
                    )
                # Reorganize the images to match the prompts
                cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                images = [
                    images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                    for i in range(len(n_images_in_text))
                ]
            else:
                images = [images]
        elif (
            not isinstance(images, (list, tuple))
            and not isinstance(images[0], (list, tuple))
            and not is_image_or_image_url(images[0][0])
        ):
            raise ValueError(
                "Invalid input images. Please provide a single image or a list of images or a list of list of images."
            )
        n_images_in_images = [len(sample) for sample in images]

        # Load images if they are URLs
        images = [[load_image(im) if is_url(im) else im for im in sample] for sample in images]

        image_inputs = self.image_processor(images, do_image_splitting=do_image_splitting, size=image_processor_size, **output_kwargs["images_kwargs"])
        inputs.update(image_inputs)

        if text is not None:
            if n_images_in_images != n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                )

            image_rows = inputs.pop("rows", [[0] * len(text)])
            image_cols = inputs.pop("cols", [[0] * len(text)])

            fake_image_token = self.fake_image_token.content
            image_token = self.image_token.content
            global_img_token = self.global_image_tag

            prompt_strings = []
            for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
                image_prompt_strings = []
                for n_rows, n_cols in zip(sample_rows, sample_cols):
                    image_prompt_string = get_image_prompt_string(
                        n_rows,
                        n_cols,
                        image_seq_len,
                        image_token=image_token,
                        fake_token_around_image=fake_image_token,
                        global_img_token=global_img_token,
                    )
                    image_prompt_strings.append(image_prompt_string)

                split_sample = sample.split(image_token)
                if len(split_sample) == 0:
                    raise ValueError("The image token should be present in the text.")

                # Place in the image prompt strings where the image tokens are
                sample = split_sample[0]
                for i, image_prompt_string in enumerate(image_prompt_strings):
                    sample += image_prompt_string + split_sample[i + 1]
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(text=prompt_strings, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

    def video_to_image_tokens(self, original_text: str, frames, timestamps, duration_sec: float) -> str:
        """
        Converts a single prompt containing <video> into a text
        with an intro, N frame placeholders, and an outro.
        """
        td = timedelta(seconds=duration_sec)
        video_intro = DEFAULT_VIDEO_INTRO.format(frame_count=num2words(len(frames)), video_duration=str(td))
        new_text = video_intro

        for i, ts in enumerate(timestamps):
            new_text += f"\n{FRAME_TIMESTAMP_MESSAGE.format(timestamp=ts)} <image>"

        new_text += f"\n{DEFAULT_MEDIA_OUTTRO}\n"
        # append whatever else the user had in the original text after <video>
        new_text += original_text
        return new_text

    def apply_chat_template(self, messages, num_frames=None, timestamps=None, duration_sec=None):
        """
        Overrides apply_chat_template to first convert any {'type': 'video'} blocks
        into a series of text+image references (video intro, frame placeholders, etc.).
        Then calls the base class apply_chat_template.

        If you already have frames/timestamps/duration, pass them in here, e.g. for
        a single video scenario. In a more general or multi-video scenario, you might
        expand this method or pass multiple sets of frames.

        This method modifies 'messages' in-place.
        """
        if frames is None or timestamps is None or duration_sec is None:
            # apply normal template
            return super().apply_chat_template(messages)
            
        # For each message, scan content for {"type": "video"}
        for msg in messages:
            if "content" not in msg:
                continue

            new_content = []
            for block in msg["content"]:
                if block.get("type") == "video":
                    assert  frames is not None or timestamps is not None or duration_sec is not None,  "to use 'video' tokens, you must specify `frames`, `timestamps`, and `duration_sec`."
                    # 1) Insert the intro
                    # frames, timestamps, duration_sec must be provided
                    # (Alternatively, you could dynamically load them here.)
                    if frames is None or timestamps is None or duration_sec is None:
                        # If user didn't pass these, raise or skip
                        raise ValueError("Must provide frames, timestamps, and duration_sec to insert 'video' blocks.")

                    # Build the video intro text
                    from datetime import timedelta
                    from num2words import num2words
                    td = timedelta(seconds=duration_sec)
                    intro_str = (
                        f"You are provided the following series of {num2words(num_frames)} frames "
                        f"from a {str(td)} [H:MM:SS] video.\n"
                    )
                    new_content.append({"type": "text", "text": intro_str})

                    # 2) Insert per-frame lines: "Frame from {timestamp}:", then an "image" block
                    for i, ts in enumerate(timestamps):
                        frame_str = f"Frame from {ts}:"
                        new_content.append({"type": "text", "text": frame_str})
                        new_content.append({"type": "image"})

                    # 3) Optionally add an outro (e.g. "Now answer the question:")
                    new_content.append({"type": "text", "text": "Now answer the following question:"})
                    # Do NOT add the original block => we skip it (since we've replaced it)
                else:
                    # keep original block
                    new_content.append(block)

            # update the content
            msg["content"] = new_content

        return super().apply_chat_template(messages)
        
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SmolVLMTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        batched_decode_output = self.tokenizer.batch_decode(*args, **kwargs)
        return [self._regex_to_remove_extra_special_tokens.sub("<image>", s) for s in batched_decode_output]

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SmolVLMTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        decode_output = self.tokenizer.decode(*args, **kwargs)
        return self._regex_to_remove_extra_special_tokens.sub("<image>", decode_output)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + tokenizer_input_names))


__all__ = ["SmolVLMProcessor"]
