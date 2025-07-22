from dotenv import load_dotenv
import tagger
import os

load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')

if __name__ == "__main__":
    tags = tagger.run(PATH_IMAGE_INPUT)
    print(f"tags: {tags}")
