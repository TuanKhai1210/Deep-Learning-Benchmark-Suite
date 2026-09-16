"""Tests for the shared A1 preprocessing and augmentation pipeline."""

from PIL import Image
import unittest

from dlbench.a1.data.transforms import build_transforms


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.preprocessing = {
            "image_size": [28, 28],
            "mean": [0.5],
            "std": [0.25],
            "augmentations": [
                {"name": "random_crop", "size": [28, 28], "padding": 2},
            ],
        }

    def test_random_crop_preserves_image_shape(self):
        transformed = build_transforms(self.preprocessing, training=True)(Image.new("L", (28, 28)))
        self.assertEqual(tuple(transformed.shape), (1, 28, 28))

    def test_evaluation_transform_ignores_training_augmentations(self):
        transformed = build_transforms(self.preprocessing, training=False)(Image.new("L", (28, 28)))
        self.assertEqual(tuple(transformed.shape), (1, 28, 28))

    def test_unknown_augmentation_is_rejected(self):
        preprocessing = {**self.preprocessing, "augmentations": [{"name": "rotate"}]}
        with self.assertRaisesRegex(ValueError, "Unsupported augmentation"):
            build_transforms(preprocessing, training=True)