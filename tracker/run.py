from dotenv import load_dotenv
import json
import os
import tracker

load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')
points = os.getenv('POSE_POINTS')
POSE_POINTS = json.loads(points) if points else {}

if __name__ == "__main__":
    image, data = tracker.run(PATH_IMAGE_INPUT, POSE_POINTS)
    tracker.save_result(PATH_IMAGE_INPUT, 'edited', image, data)
