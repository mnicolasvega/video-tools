from dotenv import load_dotenv
from pathlib import Path
import background_remover
import os

load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')

def get_output_path(input: str, label: str) -> str:
    file_path = Path(input)
    file_name = file_path.stem
    extension = file_path.suffix
    return file_path.with_name(f"{file_name} {label}{extension}")

if __name__ == "__main__":
    image = background_remover.run(PATH_IMAGE_INPUT)
    output_path = get_output_path(PATH_IMAGE_INPUT, 'nobg')
    image = image.convert('RGB')
    image.save(output_path)
