from dotenv import load_dotenv
import cv2
import json
import os
import tracker

load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')
points = os.getenv('POSE_POINTS')
POSE_POINTS = json.loads(points) if points else {}

if __name__ == "__main__":
    image_src = cv2.imread(PATH_IMAGE_INPUT)
    image, data = tracker.run(image_src, POSE_POINTS)
    tracker.save_result(PATH_IMAGE_INPUT, 'edited', image, data)
