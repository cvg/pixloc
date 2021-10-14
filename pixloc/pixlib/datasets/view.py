from pathlib import Path
import numpy as np
import cv2
# TODO: consider using PIL instead of OpenCV as it is heavy and only used here
import torch

from ..geometry import Camera, Pose


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image


def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def crop(image, size, *, random=True, other=None, camera=None,
         return_bbox=False, centroid=None):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    if random:
        top = np.random.randint(0, h - h_new + 1)
        left = np.random.randint(0, w - w_new + 1)
    elif centroid is not None:
        x, y = centroid
        top = np.clip(int(y) - h_new // 2, 0, h - h_new)
        left = np.clip(int(x) - w_new // 2, 0, w - w_new)
    else:
        top = left = 0

    image = image[top:top+h_new, left:left+w_new]
    ret = [image]
    if other is not None:
        ret += [other[top:top+h_new, left:left+w_new]]
    if camera is not None:
        ret += [camera.crop((left, top), (w_new, h_new))]
    if return_bbox:
        ret += [(top, top+h_new, left, left+w_new)]
    return ret


def zero_pad(size, *images):
    ret = []
    for image in images:
        h, w = image.shape[:2]
        padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret


def read_view(conf, image_path: Path, camera: Camera, T_w2cam: Pose,
              p3D: np.ndarray, p3D_idxs: np.ndarray, *,
              rotation=0, random=False):

    img = read_image(image_path, conf.grayscale)
    img = img.astype(np.float32)
    name = image_path.name

    # we assume that the pose and camera were already rotated during preprocess
    if rotation != 0:
        img = np.rot90(img, rotation)

    if conf.resize:
        scales = (1, 1)
        if conf.resize_by == 'max':
            img, scales = resize(img, conf.resize, fn=max)
        elif (conf.resize_by == 'min' or
                (conf.resize_by == 'min_if'
                    and min(*img.shape[:2]) < conf.resize)):
            img, scales = resize(img, conf.resize, fn=min)
        if scales != (1, 1):
            camera = camera.scale(scales)

    if conf.crop:
        if conf.optimal_crop:
            p2D, valid = camera.world2image(T_w2cam * p3D[p3D_idxs])
            p2D = p2D[valid].numpy()
            centroid = tuple(p2D.mean(0)) if len(p2D) > 0 else None
            random = False
        else:
            centroid = None
        img, camera, bbox = crop(
            img, conf.crop, random=random,
            camera=camera, return_bbox=True, centroid=centroid)
    elif conf.pad:
        img, = zero_pad(conf.pad, img)
        # we purposefully do not update the image size in the camera object

    data = {
        'name': name,
        'image': numpy_image_to_torch(img),
        'camera': camera.float(),
        'T_w2cam': T_w2cam.float(),
    }
    return data
