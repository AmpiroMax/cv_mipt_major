from image_preprocessing import (
    read_annotation,
    get_annot_path,
    get_video_path,
    frame_data_generating_preprocessing
)

import cv2
import numpy as np
from tqdm.auto import tqdm

SEGMENT_SIZE = 32
SAVE_POSITIVE_PATH = "09_motion_flow/homework_07/train_test_data/pos/"
SAVE_NEGATIVE_PATH = "09_motion_flow/homework_07/train_test_data/neg/"
DATA_CLIPS_NAMES = [
    "Clip_1",
    "Clip_4",
    "Clip_5",
    "Clip_37",
]
DATA_TEST_CLIPS_NAMES = [
    "Clip_10"
]


def generate_data():
    segment_idx = 0
    for clip_name in DATA_CLIPS_NAMES:
        annot_path = get_annot_path(clip_name)
        clip_annots = read_annotation(annot_path)

        video_path = get_video_path(clip_name)
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_annot = clip_annots[frame_idx]
                frame = frame_data_generating_preprocessing(frame)

                for annot in frame_annot:
                    strob_centre = np.array([
                        (annot[0] + annot[2]) // 2,
                        (annot[1] + annot[3]) // 2
                    ])

                    y1 = strob_centre[0] - SEGMENT_SIZE // 2
                    y2 = strob_centre[0] + SEGMENT_SIZE // 2
                    x1 = strob_centre[1] - SEGMENT_SIZE // 2
                    x2 = strob_centre[1] + SEGMENT_SIZE // 2

                    if y1 < 0 or y2 < 0 or x1 < 0 or x2 < 0:
                        continue
                    if y1 > frame.shape[0] or y2 > frame.shape[0]:
                        continue
                    if x1 > frame.shape[1] or x2 > frame.shape[1]:
                        continue

                    segment = frame[y1: y2, x1:x2, :]
                    cv2.imwrite(
                        SAVE_POSITIVE_PATH+str(segment_idx).zfill(5)+".png",
                        segment
                    )

                    if (x2 + SEGMENT_SIZE < frame.shape[1]):
                        x1 = SEGMENT_SIZE + \
                            strob_centre[1] - SEGMENT_SIZE // 2
                        x2 = SEGMENT_SIZE + \
                            strob_centre[1] + SEGMENT_SIZE // 2
                    else:
                        x1 = -SEGMENT_SIZE + \
                            strob_centre[1] - SEGMENT_SIZE // 2
                        x2 = -SEGMENT_SIZE + \
                            strob_centre[1] + SEGMENT_SIZE // 2

                    segment = frame[y1: y2, x1:x2, :]
                    cv2.imwrite(
                        SAVE_NEGATIVE_PATH+str(segment_idx).zfill(5)+".png",
                        segment
                    )

                    segment_idx += 1

                frame_idx += 1
                pbar.update(1)

        cap.release()


if __name__ == "__main__":
    generate_data()
