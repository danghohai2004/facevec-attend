import numpy as np
import pytest

from src.modules.antispoofing.service import (
    DEFAULT_MODEL_PATH,
    MiniFASNetChecker,
    PassThroughChecker,
    crop_face,
    preprocess,
)


def test_crop_face_is_square_with_padding():
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    # face bbox near the top-left corner → crop needs zero padding
    crop = crop_face(img, [0.0, 0.0, 0.2, 0.4], bbox_inc=1.5)
    assert crop.shape[0] == crop.shape[1]  # square
    assert crop.shape[0] == int(40 * 1.5)  # max(w=40px, h=40px) * 1.5
    assert (crop[0, 0] == 0).all()  # padded corner is black


def test_preprocess_shape_and_range():
    crop = np.full((60, 90, 3), 255, dtype=np.uint8)
    tensor = preprocess(crop)
    assert tensor.shape == (1, 3, 128, 128)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0 and tensor.min() >= 0.0
    # non-square input → padded rows are zero
    assert tensor[0, :, 0, :].max() == 0.0


def test_pass_through_always_true():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert PassThroughChecker().check(img, [0, 0, 1, 1]) is True


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="model not downloaded")
def test_minifasnet_runs_on_frame():
    checker = MiniFASNetChecker(DEFAULT_MODEL_PATH)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = checker.check(img, [0.3, 0.3, 0.6, 0.7])
    assert isinstance(result, bool)
    # random noise is not a real face — a sane model must reject it
    assert result is False
