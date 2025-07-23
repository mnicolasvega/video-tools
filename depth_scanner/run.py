from dotenv import load_dotenv
from pathlib import Path
import depth
import os

load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')
SAVE_FORMAT = "png"

def get_output_path(input_path: str) -> str:
    path = Path(input_path)
    output_path = path.with_name(f"{path.stem} depth.{SAVE_FORMAT}")
    return output_path

if __name__ == "__main__":
    output_path = get_output_path(PATH_IMAGE_INPUT)
    depth_map = depth.run(PATH_IMAGE_INPUT)
    image = depth.to_image(depth_map)
    image.save(output_path, format = SAVE_FORMAT)
