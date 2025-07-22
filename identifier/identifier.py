from PIL import Image
import cv2
import torch

CLASS_PERSON = 0

def get_object_bounds(input_path: str) -> list:
    img = Image.open(input_path)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(img)
    df_people = results.pandas().xyxy[CLASS_PERSON]
    person_boxes = df_people[df_people['name'] == 'person']
    bounds = []
    for _, row in person_boxes.iterrows():
        confidence = row['confidence']
        x_min, y_min, x_max, y_max = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        bounds.append([x_min, y_min, x_max, y_max, confidence])
    return bounds

def get_subimages(input_path: str, bounds: list) -> list:
    img_source = cv2.imread(input_path)
    subimages = []
    for bound in bounds:
        x_min, y_min, x_max, y_max, confidence = bound
        img_crop = img_source[y_min:y_max, x_min:x_max]
        subimages.append(img_crop)
    return subimages
