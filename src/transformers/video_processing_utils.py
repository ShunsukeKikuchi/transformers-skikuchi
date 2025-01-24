# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from .dynamic_module_utils import custom_object_save
from .image_processing_base import BatchFeature
from .image_processing_utils import BaseImageProcessor
from .image_utils import load_video
from .utils import (
    VIDEO_PROCESSOR_NAME,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    download_url,
    is_offline_mode,
    is_remote_url,
    logging,
)


logger = logging.get_logger(__name__)


INIT_SERVICE_KWARGS = [
    "processor_class",
    "video_processor_type",
]


# For now we start with slow video processor which processed each frame with image processor
# TODO: @raushan integrate video processor with torchvision to process the whole video at once
class BaseVideoProcessor(BaseImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, videos, **kwargs) -> BatchFeature:
        """Preprocess a video or a batch of videos."""
        return self.preprocess(videos, **kwargs)

    def preprocess(self, videos, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each video processor must implement its own preprocess method")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a type of [`~video_processing_utils.VideoProcessorBase`] from an video processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained video hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a video processor file saved using the
                  [`~video_processing_utils.VideoProcessorBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved video processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model video processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the video processor files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final video processor object. If `True`, then this
                functions returns a `Tuple(video_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not video processor attributes: i.e., the part of
                `kwargs` which has not been used to update `video_processor` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are video processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* video processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A video processor of type [`~video_processing_utils.ImagVideoProcessorBase`].

        Examples:

        ```python
        # We can't instantiate directly the base class *VideoProcessorBase* so let's show the examples on a
        # derived class: *LlavaOnevisionVideoProcessor*
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )  # Download video_processing_config from huggingface.co and cache.
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
            "./test/saved_model/"
        )  # E.g. video processor (or model) was saved using *save_pretrained('./test/saved_model/')*
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained("./test/saved_model/preprocessor_config.json")
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", do_normalize=False, foo=False
        )
        assert video_processor.do_normalize is False
        video_processor, unused_kwargs = LlavaOnevisionVideoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", do_normalize=False, foo=False, return_unused_kwargs=True
        )
        assert video_processor.do_normalize is False
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        video_processor_dict, kwargs = cls.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(video_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an video processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~video_processing_utils.VideoProcessorBase.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the video processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_video_processor_file = os.path.join(save_directory, VIDEO_PROCESSOR_NAME)

        self.to_json_file(output_video_processor_file)
        logger.info(f"Video processor saved in {output_video_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_video_processor_file]

    @classmethod
    def get_video_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        video processor of type [`~video_processing_utils.VideoProcessorBase`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the video processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        user_agent = {"file_type": "video processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        old_video_processor_name = "preprocessor_config.json"
        if os.path.isdir(pretrained_model_name_or_path):
            video_processor_file = os.path.join(pretrained_model_name_or_path, VIDEO_PROCESSOR_NAME)
            old_video_processor_file = os.path.join(pretrained_model_name_or_path, old_video_processor_name)
            if not os.path.exists(video_processor_file) and os.path.exists(old_video_processor_file):
                logger.warning_once(
                    "You have video processor config saved in `preprocessor.json` file which is deprecated. "
                    "Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename "
                    "the file or load and save the processor back which renames it automatically. "
                    "Loading from `preprocessor.json` will be removed in v5.0."
                )
                video_processor_file = old_video_processor_file
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_video_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            video_processor_file = pretrained_model_name_or_path
            resolved_video_processor_file = download_url(pretrained_model_name_or_path)
        else:
            try:
                # try to load with an old config name first and if not successfull try with
                # the new file name. In case we can load with old name successfully, raise a deprecation warning
                video_processor_file = old_video_processor_name
                resolved_video_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    video_processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
            except EnvironmentError:
                video_processor_file = VIDEO_PROCESSOR_NAME
                # Load from local folder or from cache or download from model Hub and cache
                resolved_video_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    video_processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load video processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {VIDEO_PROCESSOR_NAME} file"
                )
            else:
                logger.warning_once(
                    "You have video processor config saved in `preprocessor.json` file which is deprecated. "
                    "Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename "
                    "the file or load and save the processor back which renames it automatically. "
                    "Loading from `preprocessor.json` will be removed in v5.0."
                )

        try:
            # Load video_processor dict
            with open(resolved_video_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            video_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_video_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_video_processor_file}")
        else:
            logger.info(
                f"loading configuration file {video_processor_file} from cache at {resolved_video_processor_file}"
            )

        if not is_local:
            if "auto_map" in video_processor_dict:
                video_processor_dict["auto_map"] = add_model_info_to_auto_map(
                    video_processor_dict["auto_map"], pretrained_model_name_or_path
                )
            if "custom_pipelines" in video_processor_dict:
                video_processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    video_processor_dict["custom_pipelines"], pretrained_model_name_or_path
                )
        return video_processor_dict, kwargs

    @classmethod
    def from_dict(cls, video_processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~video_processing_utils.VideoProcessorBase`] from a Python dictionary of parameters.

        Args:
            video_processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the video processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~video_processing_utils.VideoProcessorBase.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the video processor object.

        Returns:
            [`~video_processing_utils.VideoProcessorBase`]: The video processor object instantiated from those
            parameters.
        """
        video_processor_dict = video_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # The `size` parameter is a dict and was previously an int or tuple in feature extractors.
        # We set `size` here directly to the `video_processor_dict` so that it is converted to the appropriate
        # dict within the video processor and isn't overwritten if `size` is passed in as a kwarg.
        if "size" in kwargs and "size" in video_processor_dict:
            video_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in video_processor_dict:
            video_processor_dict["crop_size"] = kwargs.pop("crop_size")

        video_processor = cls(**video_processor_dict)

        # Update video_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(video_processor, key):
                setattr(video_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Video processor {video_processor}")
        if return_unused_kwargs:
            return video_processor, kwargs
        else:
            return video_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this video processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["video_processor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a video processor of type [`~video_processing_utils.VideoProcessorBase`] from the path to a JSON
        file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A video processor of type [`~video_processing_utils.VideoProcessorBase`]: The video_processor object
            instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        video_processor_dict = json.loads(text)
        return cls(**video_processor_dict)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoVideoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom video processors as the ones
        in the library are already mapped with `AutoVideoProcessor `.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoVideoProcessor "`):
                The auto class to register this new video processor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def fetch_videos(self, video_url_or_urls: Union[str, List[str]]):
        """
        Convert a single or a list of urls into the corresponding `np.array` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if isinstance(video_url_or_urls, list):
            return [self.fetch_videos(x) for x in video_url_or_urls]
        elif isinstance(video_url_or_urls, str):
            return load_video(video_url_or_urls)
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(video_url_or_urls)}")


BaseVideoProcessor.push_to_hub = copy_func(BaseVideoProcessor.push_to_hub)
if BaseVideoProcessor.push_to_hub.__doc__ is not None:
    BaseVideoProcessor.push_to_hub.__doc__ = BaseVideoProcessor.push_to_hub.__doc__.format(
        object="video processor", object_class="AutoVideoProcessor", object_files="video processor file"
    )
