from rembg import remove
from PIL import Image

def run(input_path: str) -> Image:
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    return output_image.convert('RGB')
