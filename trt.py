import argparse
import os
import cv2
import torch

import torchvision.transforms.functional as F
import numpy as np

from mivolo.model.mi_volo import MiVOLO

from typing import List, Tuple, Set, Union


_IMG_SIZE: int = 640
_IMG_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser("mivolo trt")
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        type=str,
        help="path to MiVOLO checkpoint file",
    )
    parser.add_argument(
        "-d",
        "--device",
        required=True,
        type=str,
        help='CUDA device in the foramt of "cuda:0"',
    )
    parser.add_argument(
        "-s",
        "--samples",
        required=False,
        type=str,
        help="Sample input directory.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        required=True,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=False,
        type=str,
        help="TRT file output directory",
    )
    return parser


def _get_default_samples(batch_size: int) -> torch.Tensor:
    return torch.ones((batch_size, 6, _IMG_SIZE, _IMG_SIZE)).type(torch.float16)


def _get_sample_files(dir_path: str) -> List[Tuple[str, Union[None, str]]]:
    res: List[Tuple[str, Union[None, str]]] = list()
    for root, _, files in os.walk(dir_path):
        for basename in files:
            if basename.startswith("person_"):
                person_file_res: str = os.path.join(root, basename)
                face_file_res: Union[None, str] = None
                face_file: str = os.path.join(root, basename.replace("person", "face"))
                if os.path.exists(face_file) and os.path.isfile(face_file):
                    face_file_res = face_file
                res.append((person_file_res, face_file_res))
    return res


def _preproc_image(img: np.ndarray, scale_up=True) -> torch.Tensor:
    shape: Tuple[int, int] = img.shape[:2]
    new_shape: Tuple[int, int] = (_IMG_SIZE, _IMG_SIZE)

    if img.shape[0] != new_shape[0] or img.shape[1] != new_shape[1]:
        r: float = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:
            r = min(r, 1.0)
        new_unpad: Tuple[int, int] = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        color: Tuple[int, int, int] = (0, 0, 0)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = (img - _IMG_MEAN) / _IMG_STD
    img = img.astype(dtype=np.float32)

    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img


def _get_sample_inputs(
    files: List[Tuple[str, Union[None, str]]], batch_size: int, dtype=torch.float16
) -> torch.Tensor:
    assert len(files) > 0
    res: Union[None, torch.Tensor] = None
    for person_f, face_f in files:
        person: torch.Tensor = _preproc_image(cv2.imread(person_f))
        face: Union[None, torch.Tensor] = None
        if face_f is None:
            face = torch.zeros((3, _IMG_SIZE, _IMG_SIZE), dtype=torch.float32)
            img = F.normalize(img, mean=_IMG_STD, std=_IMG_STD)
            img = img.unsqueeze(0)
        else:
            face: np.ndarray = cv2.imread(face_f)
            face = torch.permute(face, (2, 0, 1))
        sample: torch.Tensor = torch.cat((person, face), dim=1)
        if res is None:
            res = sample
        else:
            res = torch.cat((res, sample), dim=0)
    if res.shape[0] >= batch_size:
        return res[:batch_size].type(dtype)
    size_diff: int = batch_size - res.shape[0]
    res = torch.cat((res, res[:size_diff]), dim=0)
    return res


if __name__ == "__main__":
    parser: argparse.ArgumentParser = _build_arg_parser()
    args = parser.parse_args()

    model = MiVOLO(
        args.checkpoint,
        args.device,
        half=True,
        use_persons=True,
        disable_faces=False,
        verbose=False,
        torchcompile=None,
    )

    samples: np.ndarray = _get_default_samples(args.batch_size)
    if args.samples is not None and len(args.samples) > 0:
        sample_files: List[Tuple[str, Union[None, str]]] = _get_sample_files(
            args.samples
        )
        samples = _get_sample_inputs(sample_files, args.batch_size)
