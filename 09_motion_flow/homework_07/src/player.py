""" HW9 python script """

import cv2
from image_preprocessing import (
    get_annot_path,
    get_video_path,
    read_annotation,
    frame_preprocessing,
    draw_strob
)


def play_video(video_name: str) -> None:
    video_path = get_video_path(video_name)
    cap = cv2.VideoCapture(video_path)

    annot_path = get_annot_path(video_name)
    annots = read_annotation(annot_path)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = frame_preprocessing(frame)
        frame = draw_strob(frame, annots[frame_idx])
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_name = "Clip_4"
    play_video(video_name)
