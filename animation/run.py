from dotenv import load_dotenv
import cv2
import effect.zoom
import os

load_dotenv()
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_VIDEO_PATH = os.getenv('IMAGE_OUTPUT_PATH')
RESOLUTION = (640, 480)
NUM_FRAMES = 60 * 5
ZOOM_STRENGTH = 0.08 * 5
MIDAS_MODEL_TYPE = "MiDaS_small"  # "DPT_Large"

def run() -> None:
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"could not read: {IMAGE_PATH}")
        return

    model, transform = effect.zoom.load_midas_model()
    print("performing depth scan")
    depth = effect.zoom.estimate_depth(image, model, transform)
    print("animating")
    frames = effect.zoom.generate_zoom_frames(image, depth, NUM_FRAMES, ZOOM_STRENGTH)
    print(f"saving: {OUTPUT_VIDEO_PATH}")
    effect.zoom.save_video(frames, OUTPUT_VIDEO_PATH)
    print("saved")

if __name__ == "__main__":
    run()
