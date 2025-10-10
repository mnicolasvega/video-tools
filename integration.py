from background_remover import background_remover
from datetime import datetime
from dotenv import load_dotenv
from identifier import identifier
from pathlib import Path
import cv2
import os


load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')


def get_output_path(input: str, label: str) -> str:
    file_path = Path(input)
    file_name = file_path.stem
    extension = file_path.suffix
    timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
    return file_path.with_name(f"{file_name} {label} {timestamp}{extension}")


if __name__ == "__main__":
    bounds = identifier.get_object_bounds(PATH_IMAGE_INPUT)
    image = cv2.imread(PATH_IMAGE_INPUT)
    image_w_bounds = image.copy()
    identifier.draw_bounds(image_w_bounds, bounds)
    cv2.imwrite(get_output_path(PATH_IMAGE_INPUT, " bounds"), image_w_bounds)
    subimages = identifier.get_subimages(image, bounds, 0)
    i = 1
    for image in subimages:
        output_path = get_output_path(PATH_IMAGE_INPUT, f" person {i}")
        cv2.imwrite(output_path, image)

        output_path_no_bg = get_output_path(PATH_IMAGE_INPUT, f" person {i} nobg")
        image_no_bg = background_remover.run(output_path)
        image_no_bg.save(output_path_no_bg)
        i = i + 1
