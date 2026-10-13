import pytest
import torch

from caustics_backscatter.registry import PIXEL_WEIGHTERS
from caustics_backscatter.weighting import SegmentationWeighter, UniformWeighter


def test_segmentation_weighter_zeroes_classes():
    weighter = PIXEL_WEIGHTERS.build({"name": "segmentation", "zero_classes": [7, 8, 9, 15]})
    labels = torch.arange(40, dtype=torch.uint8).view(1, 1, 1, 4, 10).expand(2, 3, 1, 4, 10)
    images = torch.rand(2, 3, 3, 4, 10)
    weights = weighter(images, labels)
    assert weights.shape == (2, 3, 1, 4, 10) and weights.dtype == images.dtype
    zeroed = torch.isin(labels, torch.tensor([7, 8, 9, 15], dtype=torch.uint8))
    assert torch.equal(weights == 0, zeroed)
    assert torch.all(weights[~zeroed] == 1)


def test_segmentation_weighter_needs_labels():
    with pytest.raises(ValueError, match="data.labels"):
        SegmentationWeighter([9])(torch.rand(1, 2, 3, 4, 4))
    with pytest.raises(ValueError):
        SegmentationWeighter([40])


def test_uniform_weighter():
    images = torch.rand(2, 2, 3, 4, 4)
    assert torch.equal(UniformWeighter()(images), torch.ones(2, 2, 1, 4, 4))
    assert torch.equal(UniformWeighter()(images, torch.zeros(2, 2, 1, 4, 4, dtype=torch.uint8)), torch.ones(2, 2, 1, 4, 4))
