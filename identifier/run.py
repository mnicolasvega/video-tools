from dotenv import load_dotenv
from pathlib import Path
import cv2
import identifier
import os


load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')


def get_output_path(input: str, label: str) -> str:
    file_path = Path(input)
    file_name = file_path.stem
    extension = file_path.suffix
    return file_path.with_name(f"{file_name} {label}{extension}")


if __name__ == "__main__":
    bounds = identifier.get_object_bounds(PATH_IMAGE_INPUT)
    subimages = identifier.get_subimages(PATH_IMAGE_INPUT, bounds)
    i = 1
    for image in subimages:
        output_path = get_output_path(PATH_IMAGE_INPUT, f" person {i}")
        cv2.imwrite(output_path, image)
        i = i + 1
