import unittest

import numpy as np
import requests
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import EoMTImageProcessor

    if is_torchvision_available():
        from transformers import EoMTImageProcessorFast
    from transformers.models.eomt.modeling_eomt import EoMTForUniversalSegmentationOutput


class EoMTImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        size=None,
        do_resize=True,
        do_pad=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        num_labels=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_pad = do_pad
        self.size = size if size is not None else {"shortest_edge": 18, "longest_edge": 18}
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        # for the post_process_functions
        self.batch_size = 2
        self.num_queries = 3
        self.num_classes = 2
        self.height = 18
        self.width = 18
        self.num_labels = num_labels

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
            "num_labels": self.num_labels,
        }

    def prepare_fake_eomt_outputs(self, batch_size):
        return EoMTForUniversalSegmentationOutput(
            masks_queries_logits=torch.randn((batch_size, self.num_queries, self.height, self.width)),
            class_queries_logits=torch.randn((batch_size, self.num_queries, self.num_classes + 1)),
        )

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


def prepare_semantic_single_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test", trust_remote_code=True)

    image = Image.open(dataset[0]["file"])
    map = Image.open(dataset[1]["file"])

    return image, map


def prepare_semantic_batch_inputs():
    ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test", trust_remote_code=True)

    image1 = Image.open(ds[0]["file"])
    map1 = Image.open(ds[1]["file"])
    image2 = Image.open(ds[2]["file"])
    map2 = Image.open(ds[3]["file"])

    return [image1, image2], [map1, map2]


@require_torch
@require_vision
class EoMTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = EoMTImageProcessor if is_vision_available() else None
    fast_image_processing_class = EoMTImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = EoMTImageProcessingTester(self)
        self.model_id = "tue-mps/coco_panoptic_eomt_large_640"

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "resample"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 18, "longest_edge": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"shortest_edge": 42})

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (2, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass

    def test_call_pil(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test Non batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (1, 3, 18, 18)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (2, 3, 18, 18)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (1, 3, 18, 18)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (2, 3, 18, 18)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image, dummy_map = prepare_semantic_single_inputs()

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        image_encoding_slow = image_processor_slow(dummy_image, segmentation_maps=dummy_map, return_tensors="pt")
        image_encoding_fast = image_processor_fast(dummy_image, segmentation_maps=dummy_map, return_tensors="pt")

        self.assertTrue(torch.allclose(image_encoding_slow.pixel_values, image_encoding_fast.pixel_values, atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(image_encoding_slow.pixel_values - image_encoding_fast.pixel_values)).item(), 1e-3
        )

    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images, dummy_maps = prepare_semantic_batch_inputs()

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, segmentation_maps=dummy_maps, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, segmentation_maps=dummy_maps, return_tensors="pt")

        self.assertTrue(torch.allclose(encoding_slow.pixel_values, encoding_fast.pixel_values, atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow.pixel_values - encoding_fast.pixel_values)).item(), 1e-3
        )

    def test_post_process_semantic_segmentation(self):
        processor = self.image_processing_class(**self.image_processor_dict)
        # Set longest_edge to None to test for semantic segmentatiom.
        processor.size = {"shortest_edge": 18, "longest_edge": None}
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, do_split_image=True, return_tensors="pt")
        patch_offsets = inputs.pop("patch_offsets")

        original_sizes = [image.size[::-1]]

        # For semantic segmentation, the BS of output is 2 coz, two patches are created for the image.
        outputs = self.image_processor_tester.prepare_fake_eomt_outputs(inputs["pixel_values"].shape[0])
        segmentation = processor.post_process_semantic_segmentation(outputs, patch_offsets, original_sizes)

        self.assertEqual(segmentation[0].shape, (image.height, image.width))

    def test_post_process_panoptic_segmentation(self):
        processor = self.image_processing_class(**self.image_processor_dict)
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        original_sizes = [image.size[::-1], image.size[::-1]]

        # lets test for batched input of 2
        outputs = self.image_processor_tester.prepare_fake_eomt_outputs(2)
        segmentation = processor.post_process_panoptic_segmentation(outputs, original_sizes)

        self.assertTrue(len(segmentation) == 2)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(el["segmentation"].shape, (image.height, image.width))

    def test_post_process_instance_segmentation(self):
        processor = self.image_processing_class(**self.image_processor_dict)
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        original_sizes = [image.size[::-1], image.size[::-1]]

        # lets test for batched input of 2
        outputs = self.image_processor_tester.prepare_fake_eomt_outputs(2)
        segmentation = processor.post_process_instance_segmentation(outputs, original_sizes)

        self.assertTrue(len(segmentation) == 2)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(el["segmentation"].shape, (image.height, image.width))
