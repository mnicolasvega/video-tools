from numpy import ndarray
from PIL import Image
import cv2
import torch

CLASS_PERSON = 0
REPOSITORY = 'ultralytics/yolov5'
MODEL = 'yolov5s'

def get_model():
    return torch.hub.load(REPOSITORY, MODEL, pretrained = True)

def get_object_bounds(input_path: str, _tag: str = 'person', _class: int = CLASS_PERSON, model = None) -> list:
    img = Image.open(input_path)
    if model == None:
        model = get_model()
    results = model(img)
    df_people = results.pandas().xyxy[_class]
    person_boxes = df_people[df_people['name'] == _tag]
    bounds = []
    for _, row in person_boxes.iterrows():
        confidence = row['confidence']
        x_min, y_min, x_max, y_max = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        bounds.append([x_min, y_min, x_max, y_max, confidence])
    img.close()
    results = None
    return bounds

def get_subimages(img: ndarray, bounds: list) -> list:
    subimages = []
    for bound in bounds:
        x_min, y_min, x_max, y_max, confidence = bound
        img_crop = img[y_min:y_max, x_min:x_max]
        subimages.append(img_crop)
    return subimages

def draw_bounds(img: ndarray, bounds: list, color: tuple = (128, 128, 255), thickness: int = 3) -> None:
    for bound in bounds:
        x_min, y_min, x_max, y_max, confidence = bound
        color = (128, 128, 255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
