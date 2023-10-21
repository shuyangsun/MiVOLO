import argparse
import time
import os
import cv2
import torch

import numpy as np
import tensorrt as trt
import torchvision.transforms.functional as F

from mivolo.model.mi_volo import MiVOLO
from torch2trt import torch2trt

from typing import List, Tuple, Set, Union


_IMG_SIZE: int = 224
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
        "--inputs",
        required=False,
        type=str,
        help="Sample input directory.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        required=True,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "-w", "--workspace", type=int, default=32, help="max workspace size in detect"
    )
    parser.add_argument(
        "-i",
        "--iters",
        required=False,
        type=int,
        default=512,
        help="number of iterations on samples to test performance",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=False,
        type=str,
        help="TRT file output directory",
    )
    return parser


def _get_default_samples(batch: int) -> torch.Tensor:
    return torch.ones((batch, 6, _IMG_SIZE, _IMG_SIZE)).half()


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
    return img.half()


def _get_sample_inputs(
    files: List[Tuple[str, Union[None, str]]], batch: int
) -> torch.Tensor:
    assert len(files) > 0
    res: Union[None, torch.Tensor] = None
    for person_f, face_f in files:
        person: torch.Tensor = _preproc_image(cv2.imread(person_f))
        face: Union[None, torch.Tensor] = None
        if face_f is None:
            face = torch.zeros((3, _IMG_SIZE, _IMG_SIZE), dtype=torch.float32)
            face = F.normalize(face, mean=_IMG_STD, std=_IMG_STD)
            face = face.unsqueeze(0)
        else:
            face = _preproc_image(cv2.imread(face_f))
        sample: torch.Tensor = torch.cat((person, face), dim=1)
        if res is None:
            res = sample
        else:
            res = torch.cat((res, sample), dim=0)
    while res.shape[0] < batch:
        size_diff: int = batch - res.shape[0]
        res = torch.cat((res, res[:size_diff]), dim=0)
    return res.half()


@torch.no_grad()
def main():
    parser: argparse.ArgumentParser = _build_arg_parser()
    args = parser.parse_args()

    with torch.cuda.device(args.device):
        model = MiVOLO(
            args.checkpoint,
            args.device,
            half=True,
            use_persons=True,
            disable_faces=False,
            verbose=False,
            torchcompile=None,
        ).model

        inputs: torch.Tensor = _get_default_samples(args.batch)
        if args.inputs is not None and len(args.inputs) > 0:
            sample_files: List[Tuple[str, Union[None, str]]] = _get_sample_files(
                args.inputs
            )
            inputs = _get_sample_inputs(sample_files, args.batch)
        inputs = inputs.to(args.device)
        inputs = [inputs]

        num_frames = int((((args.iters - 1) // args.batch) + 1) * args.batch)
        start = time.time()
        for _ in range(num_frames // args.batch):
            with torch.no_grad():
                pred = model(inputs[0])

        print(
            "PyTorch model fps (avg of {num} samples): {fps:.1f}".format(
                num=num_frames, fps=num_frames / (time.time() - start)
            )
        )

        model_trt = torch2trt(
            model,
            inputs,
            fp16_mode=True,
            log_level=trt.Logger.INFO,
            max_workspace_size=(1 << args.workspace),
            max_batch_size=args.batch,
        )

        # model(inputs[0]) # populate model.head
        start = time.time()
        for _ in range(num_frames // args.batch):
            pred = model_trt(inputs[0])
            # model.head.decode_outputs(pred, dtype=torch.float16, device="cuda:0")
        print(
            "TensorRT model fps (avg of {num} inputs): {fps:.1f}".format(
                num=num_frames, fps=num_frames / (time.time() - start)
            )
        )

        device_postfix: str = args.device.replace(":", "")
        torch.save(
            model_trt.state_dict(),
            os.path.join(args.out, f"mivolo_trt_b{args.batch}_{device_postfix}.pth"),
        )

        print("Converted TensorRT model done.")
        engine_file = os.path.join(
            args.out, f"mivolo_trt_b{args.batch}_{device_postfix}.engine"
        )
        with open(engine_file, "wb") as f:
            f.write(model_trt.engine.serialize())

        print("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
