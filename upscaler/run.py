from dotenv import load_dotenv
from pathlib import Path
import cv2
import os
import upscaler_cpu

load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')
SAVE_FORMAT = "png"
SCALE_FACTOR = 4

def get_output_path(input_path: str) -> str:
    path = Path(input_path)
    output_path = path.with_name(f"{path.stem} upscaled x{SCALE_FACTOR}.{SAVE_FORMAT}")
    return output_path

if __name__ == "__main__":
    path_output = get_output_path(PATH_IMAGE_INPUT)
    img_upscaled = upscaler_cpu.run(PATH_IMAGE_INPUT, SCALE_FACTOR)
    cv2.imwrite(path_output, img_upscaled)
