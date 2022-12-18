from player import play_video
from train_classificator_model import (
    SimplePerceptron,
    train_model
)
from train_data_generator import generate_data

if __name__ == "__main__":
    video_name = "Clip_1"

    generate_data("train")
    generate_data("test")

    train_model()

    play_video(video_name)
