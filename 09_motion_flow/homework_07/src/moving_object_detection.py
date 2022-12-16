import cv2
import numpy as np
from typing import List
import torch

from train_data_generator import (
    SEGMENT_SIZE
)
from train_classificator_model import (
    DEVICE,
    dataset_image_preprocessing,

)


def get_canditates_centres(frame: cv2.Mat) -> List[np.ndarray]:
    coords = []
    corners = cv2.goodFeaturesToTrack(
        image=frame,
        maxCorners=20,
        qualityLevel=1e-6,
        minDistance=30,
        useHarrisDetector=True
    )
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        coords += [np.array([x, y])]

    return coords


def create_canditates_annotations(frame: cv2.Mat) -> List[np.ndarray]:
    centres = get_canditates_centres(frame)
    annots = []

    for x, y in centres:
        y1 = y - SEGMENT_SIZE // 2
        y2 = y + SEGMENT_SIZE // 2
        x1 = x - SEGMENT_SIZE // 2
        x2 = x + SEGMENT_SIZE // 2

        if y1 < 0 or y2 < 0 or x1 < 0 or x2 < 0:
            continue
        if y1 > frame.shape[0] or y2 > frame.shape[0]:
            continue
        if x1 > frame.shape[1] or x2 > frame.shape[1]:
            continue

        annots += [np.array([y1, x1, y2, x2])]
    return annots


def predict_strobs(frame: cv2.Mat, model: torch.nn.Module) -> List[np.ndarray]:
    centres = get_canditates_centres(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    annots = []

    for x, y in centres:
        y1 = y - SEGMENT_SIZE // 2
        y2 = y + SEGMENT_SIZE // 2
        x1 = x - SEGMENT_SIZE // 2
        x2 = x + SEGMENT_SIZE // 2

        if y1 < 0 or y2 < 0 or x1 < 0 or x2 < 0:
            continue
        if y1 > frame.shape[0] or y2 > frame.shape[0]:
            continue
        if x1 > frame.shape[1] or x2 > frame.shape[1]:
            continue

        segment = frame[y1: y2, x1:x2, :]
        segment = dataset_image_preprocessing(segment).to(DEVICE)
        segment = torch.flatten(segment)[None, ...]
        prediction = model(segment).cpu()
        if prediction < 0.6:
            continue
        annots += [np.array([y1, x1, y2, x2])]

    return annots
