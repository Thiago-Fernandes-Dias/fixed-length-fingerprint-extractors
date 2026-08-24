from typing import Union

import numpy as np
import torch
import torchvision.transforms.functional as VTF

from flx.setup.config import INPUT_SIZE


def get_input_resolution() -> tuple[int, int]:
    return (INPUT_SIZE, INPUT_SIZE)


def pad_and_resize(
    img: Union[np.ndarray, torch.Tensor],
    target_size: tuple[int, int] = None,
    fill: float = 0.0,
) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = VTF.to_tensor(img)

    height = img.shape[1]
    width = img.shape[2]
    pad_width = 0 if width >= height else int((height - width) / 2)
    pad_height = 0 if height >= width else int((width - height) / 2)
    img = VTF.pad(
        img, padding=(pad_width, pad_height, pad_width, pad_height), fill=fill
    )  # left, top, right, bottom

    assert img.shape[1] == img.shape[2]

    return VTF.resize(img, target_size, antialias=True)


def pad_and_resize_to_deepprint_input_size(
    img: Union[np.ndarray, torch.Tensor],
    roi: Union[None, tuple[int, int]] = None,
    fill: float = 0.0,
) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = VTF.to_tensor(img)

    if roi is not None:
        img = VTF.center_crop(img, roi)

    return pad_and_resize(img, (INPUT_SIZE, INPUT_SIZE), fill=fill)


def transform_to_input_size(
    minutia_points: np.ndarray,
    original_height: int,
    original_width: int,
    roi: Union[None, tuple[int, int]] = None,
) -> np.ndarray:
    """
    Transforms the pixel coordinates in the same way that the pixels in the original image would be
    transformed by pad_and_resize_to_deepprint_input_size.
    """
    minutia_points = minutia_points.astype(np.float16)

    if minutia_points.shape[0] == 0:
        return minutia_points

    if roi is not None:
        minutia_points -= np.array(
            [(original_width - roi[1]) / 2, (original_height - roi[0]) / 2]
        )
        height = roi[0]
        width = roi[1]
    else:
        height = original_height
        width = original_width

import cv2


def affine_matrix(scale=1.0, theta=0.0, trans=np.zeros(2), trans_2=np.zeros(2)):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    t = np.dot(R, trans) + trans_2
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])


def coarse_center(img: np.ndarray, img_ppi: int = 500) -> np.ndarray:
    img_uint = np.rint(img).astype(np.uint8)
    ksize1 = int(19 * img_ppi / 500)
    ksize2 = int(5 * img_ppi / 500)
    seg = cv2.GaussianBlur(img_uint, ksize=(ksize1, ksize1), sigmaX=0, borderType=cv2.BORDER_REPLICATE)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2)))
    seg = seg.astype(np.float32)

    grid = np.stack(np.meshgrid(*[np.arange(x) for x in img.shape[:2]], indexing="ij")).reshape(2, -1)
    img_c = (seg.reshape(1, -1) * grid).sum(1) / seg.sum().clip(1e-6, None)
    return img_c


def flare_image_transform(
    img: np.ndarray | torch.Tensor,
    tar_shape: tuple[int, int] = (256, 256),
    middle_shape: tuple[int, int] = (256, 256),
    pose_2d: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Standard FLARE/FDD image preprocessing: affine normalization using pose or coarse center,
    rescaling pixel values to [-1, 1], output shape [1, H, W].
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()

    tar_shape_arr = np.array(tar_shape)
    middle_shape_arr = np.array(middle_shape)

    center = tar_shape_arr[::-1] / 2.0
    shift = np.zeros(2)
    if pose_2d is not None:
        img_c = pose_2d[:2]
        theta = pose_2d[2]
    else:
        img_c = coarse_center(img, img_ppi=500)[::-1]
        theta = 0

    T = affine_matrix(
        scale=tar_shape_arr[0] * 1.0 / middle_shape_arr[0],
        theta=np.deg2rad(theta),
        trans=-img_c,
        trans_2=center + shift,
    )

    img_warped = cv2.warpAffine(
        (img.astype(np.float32) - 127.5) / 127.5,
        T[:2],
        dsize=tuple(tar_shape_arr[::-1]),
        flags=cv2.INTER_LINEAR,
    )

    return torch.from_numpy(img_warped[None].astype(np.float32))

