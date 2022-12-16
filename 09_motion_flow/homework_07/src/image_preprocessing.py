import cv2
import numpy as np
from typing import List
from ast import literal_eval as make_tuple

SCALING_FACTOR = 2
ANNOTS_PATH = "09_motion_flow/homework_07/motion_detection/Annotation/"
VIDOES_PATH = "09_motion_flow/homework_07/motion_detection/Videos/"


def get_annot_path(video_name: str) -> str:
    return ANNOTS_PATH + video_name + "_gt.txt"


def get_video_path(video_name: str) -> str:
    return VIDOES_PATH + video_name + ".mov"


def frame_detector_preprocessing(
    frame: cv2.Mat
) -> cv2.Mat:
    new_w = frame.shape[1] // SCALING_FACTOR
    new_h = frame.shape[0] // SCALING_FACTOR
    frame = cv2.resize(frame, (new_w, new_h))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def frame_data_generating_preprocessing(
    frame: cv2.Mat
) -> cv2.Mat:
    new_w = frame.shape[1] // SCALING_FACTOR
    new_h = frame.shape[0] // SCALING_FACTOR
    frame = cv2.resize(frame, (new_w, new_h))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def frame_preprocessing(frame: cv2.Mat) -> cv2.Mat:
    new_w = frame.shape[1] // SCALING_FACTOR
    new_h = frame.shape[0] // SCALING_FACTOR
    frame = cv2.resize(frame, (new_w, new_h))
    return frame


def read_annotation(annot_path: str) -> List[List[np.ndarray]]:
    annots = []

    with open(annot_path) as file:
        for line in file:
            coords = line.split("detections: ")[1]
            coords = coords.split("), ")[:-1]

            frame_detections = []
            for idx in range(len(coords)):
                curr_coord = coords[idx] + ")"
                frame_detections += [
                    np.array(make_tuple(curr_coord)) // SCALING_FACTOR
                ]
            annots += [frame_detections]

    return annots


def draw_strob(frame: cv2.Mat, frame_annotations: List[np.ndarray]) -> cv2.Mat:
    for annot in frame_annotations:
        cv2.rectangle(
            frame,
            pt1=(annot[1], annot[0]),
            pt2=(annot[3], annot[2]),
            color=0,
            thickness=1
        )
    return frame
