from image_preprocessing import (
    read_annotation,
    get_annot_path,
    get_video_path,
    frame_data_generating_preprocessing
)

import cv2
import logging
import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(level="INFO")
logger = logging.getLogger("DATA_GENERATOR")

SEGMENT_SIZE = 16
SAVE_TRAIN_POSITIVE_PATH = "09_motion_flow/homework_07/train_test_data/train/pos/"
SAVE_TRAIN_NEGATIVE_PATH = "09_motion_flow/homework_07/train_test_data/train/neg/"

SAVE_TEST_POSITIVE_PATH = "09_motion_flow/homework_07/train_test_data/test/pos/"
SAVE_TEST_NEGATIVE_PATH = "09_motion_flow/homework_07/train_test_data/test/neg/"

DATA_TRAIN_CLIPS_NAMES = [
    "Clip_1",
    "Clip_4",
    "Clip_5",
    "Clip_37",
]
DATA_TEST_CLIPS_NAMES = [
    "Clip_2",
    "Clip_7",
    "Clip_10",
    "Clip_11",
]


def generate_data(mode: str = "train"):
    logger.info(f"Generating data for {mode}")
    segment_idx = 0

    segments = []

    clips_names = DATA_TRAIN_CLIPS_NAMES
    save_neg_path = SAVE_TRAIN_NEGATIVE_PATH
    save_pos_path = SAVE_TRAIN_POSITIVE_PATH
    if mode == "test":
        clips_names = DATA_TEST_CLIPS_NAMES
        save_neg_path = SAVE_TEST_NEGATIVE_PATH
        save_pos_path = SAVE_TEST_POSITIVE_PATH

    for clip_name in clips_names:
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

                    segment = frame[y1: y2, x1:x2]
                    segments += [segment]

                    cv2.imwrite(
                        save_pos_path+str(segment_idx).zfill(5)+".png",
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

                    segment = frame[y1: y2, x1:x2]
                    segments += [segment]
                    cv2.imwrite(
                        save_neg_path+str(segment_idx).zfill(5)+".png",
                        segment
                    )

                    segment_idx += 1

                frame_idx += 1
                pbar.update(1)

        cap.release()
        cv2.destroyAllWindows()
    mean = np.mean(np.array(segments), axis=(0, 1, 2))
    std = np.std(np.array(segments), axis=(0, 1, 2))

    logger.info(f"Data mean {mean}")
    logger.info(f"Data std  {std}")
    logger.info("Data was generated")


if __name__ == "__main__":
    generate_data("train")
    generate_data("test")
