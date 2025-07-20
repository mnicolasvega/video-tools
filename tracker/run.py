from dotenv import load_dotenv
from tracker import run, save_result
import os


load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')


if __name__ == "__main__":
    image, data = run(PATH_IMAGE_INPUT)
    save_result(PATH_IMAGE_INPUT, 'edited', image, data)
