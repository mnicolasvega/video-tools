from numpy import ndarray
from parts_provider import get_pose_points
from pathlib import Path
import cv2
import json
import mediapipe as mp
import os



CONFIG_DRAW_ALL_LANDMARKS_FOUND = False
CONFIG_DRAW_SPECIFIED_LANDMARKS_FOUND = True
CONFIG_DRAW_SPECIFIED_LANDMARKS_FOUND_NAMES = False



def save_result(input_path: str, image: ndarray, data: dict) -> None:
    file_name = Path(input_path)
    output_file_base = file_name.with_name(f"{file_name.stem}_edited")
    output_file_image = f"{output_file_base}{file_name.suffix}"
    output_file_json = f"{output_file_base}.json"
    cv2.imwrite(output_file_image, image)

    with open(output_file_json, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def run(input_path: str):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: '{input_path}'")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(input_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    height, width, _ = image.shape
    data = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if CONFIG_DRAW_ALL_LANDMARKS_FOUND:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        points = get_pose_points()
        ignored_points = []

        for point_name, point in points.items():
            landmark = landmarks[point]
            if landmark.visibility > 0.5:
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                if CONFIG_DRAW_SPECIFIED_LANDMARKS_FOUND:
                    cv2.circle(image, (x_px, y_px), 10, (255, 0, 0), -1)
                if CONFIG_DRAW_SPECIFIED_LANDMARKS_FOUND_NAMES:
                    cv2.putText(image, point_name, (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                data[point_name] = {
                    'x': x_px,
                    'y': y_px,
                    'confidence': round(landmark.visibility, 3)
                }
            else:
                ignored_points.append(point_name)

        if len(ignored_points) > 0:
            print("points not found: " + str(ignored_points))
    return image, data
