""" HW9 python script """

from image_preprocessing import (
    get_annot_path,
    get_video_path,
    read_annotation,
    frame_preprocessing,
    draw_strob
)
from train_classificator_model import (
    MODEL_SAVE_PATH,
    DEVICE,
    SimplePerceptron
)
from moving_object_detection import (
    predict_strobs
)
import cv2
import torch


def play_video(video_name: str) -> None:
    video_path = get_video_path(video_name)
    cap = cv2.VideoCapture(video_path)

    model = torch.load(MODEL_SAVE_PATH+"model").to(DEVICE)
    model.eval()

    annot_path = get_annot_path(video_name)
    annots = read_annotation(annot_path)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_preprocessing(frame)
        predicted_annots = predict_strobs(frame, model)
        frame = draw_strob(frame, predicted_annots)
        frame = draw_strob(frame, annots[frame_idx])

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_name = "Clip_1"
    play_video(video_name)
