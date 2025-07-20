from dotenv import load_dotenv
from numpy import ndarray
from pathlib import Path
import cv2
import json
import mediapipe as mp
import os



load_dotenv()
PATH_IMAGE_INPUT = os.getenv('PATH_IMAGE_INPUT')
CONFIG_DRAW_ALL_LANDMARKS_FOUND = False
CONFIG_DRAW_SPECIFIED_LANDMARKS_FOUND = True
CONFIG_DRAW_SPECIFIED_LANDMARKS_FOUND_NAMES = False



def get_points(
    draw_face: bool = True,        
    draw_trunk: bool = True,
    draw_arms: bool = True,
    draw_legs: bool = True,
    draw_hands: bool = False,
    draw_feet: bool = False,
) -> dict:
    pose = mp.solutions.pose
    POINTS_FACE = {
        'NOSE':           pose.PoseLandmark.NOSE,
        'EYE_INNER (L)':  pose.PoseLandmark.LEFT_EYE_INNER,
        'EYE (L)':        pose.PoseLandmark.LEFT_EYE,
        'EYE_OUTER (L)':  pose.PoseLandmark.LEFT_EYE_OUTER,
        'EYE_INNER (R)':  pose.PoseLandmark.RIGHT_EYE_INNER,
        'EYE (R)':        pose.PoseLandmark.RIGHT_EYE,
        'EYE_OUTER (R)':  pose.PoseLandmark.RIGHT_EYE_OUTER,
        'EAR (L)':        pose.PoseLandmark.LEFT_EAR,
        'EAR (R)':        pose.PoseLandmark.RIGHT_EAR,
        'MOUTH (L)':      pose.PoseLandmark.MOUTH_LEFT,
        'MOUTH (R)':      pose.PoseLandmark.MOUTH_RIGHT,
    }
    POINTS_TRUNK = {
        'SHOULDER (L)':   pose.PoseLandmark.LEFT_SHOULDER,
        'SHOULDER (R)':   pose.PoseLandmark.RIGHT_SHOULDER,
        'HIP (L)':        pose.PoseLandmark.LEFT_HIP,
        'HIP (R)':        pose.PoseLandmark.RIGHT_HIP,
    }
    POINTS_ARMS = {
        'SHOULDER (L)':   pose.PoseLandmark.LEFT_SHOULDER,
        'SHOULDER (R)':   pose.PoseLandmark.RIGHT_SHOULDER,
        'ELBOW (L)':      pose.PoseLandmark.LEFT_ELBOW,
        'ELBOW (R)':      pose.PoseLandmark.RIGHT_ELBOW,
        'WRIST (L)':      pose.PoseLandmark.LEFT_WRIST,
        'WRIST (R)':      pose.PoseLandmark.RIGHT_WRIST,
    }
    POINTS_LEGS = {
        'HIP (L)':        pose.PoseLandmark.LEFT_HIP,
        'HIP (R)':        pose.PoseLandmark.RIGHT_HIP,
        'KNEE (L)':       pose.PoseLandmark.LEFT_KNEE,
        'KNEE (R)':       pose.PoseLandmark.RIGHT_KNEE,
        'ANKLE (L)':      pose.PoseLandmark.LEFT_ANKLE,
        'ANKLE (R)':      pose.PoseLandmark.RIGHT_ANKLE,
    }
    POINTS_HANDS = {
        'WRIST (L)':      pose.PoseLandmark.LEFT_WRIST,
        'WRIST (R)':      pose.PoseLandmark.RIGHT_WRIST,
        'PINKY (L)':      pose.PoseLandmark.LEFT_PINKY,
        'PINKY (R)':      pose.PoseLandmark.RIGHT_PINKY,
        'THUMB (L)':      pose.PoseLandmark.LEFT_THUMB,
        'THUMB (R)':      pose.PoseLandmark.RIGHT_THUMB,
        'INDEX (L)':      pose.PoseLandmark.LEFT_INDEX,
        'INDEX (R)':      pose.PoseLandmark.RIGHT_INDEX,
    }
    POINTS_FEET = {
        'ANKLE (L)':      pose.PoseLandmark.LEFT_ANKLE,
        'ANKLE (R)':      pose.PoseLandmark.RIGHT_ANKLE,
        'HEEL (L)':       pose.PoseLandmark.LEFT_HEEL,
        'HEEL (R)':       pose.PoseLandmark.RIGHT_HEEL,
        'FOOT_INDEX (L)': pose.PoseLandmark.LEFT_FOOT_INDEX,
        'FOOT_INDEX (R)': pose.PoseLandmark.RIGHT_FOOT_INDEX
    }
    parts = [
        POINTS_FACE if draw_face else {},
        POINTS_TRUNK if draw_trunk else {},
        POINTS_ARMS if draw_arms else {},
        POINTS_LEGS if draw_legs else {},
        POINTS_HANDS if draw_hands else {},
        POINTS_FEET if draw_feet else {},
    ]
    all_parts = {}
    for part in parts:
        all_parts.update(part)
    return all_parts



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
        points = get_points()
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



image, data = run(PATH_IMAGE_INPUT)
save_result(PATH_IMAGE_INPUT, image, data)
