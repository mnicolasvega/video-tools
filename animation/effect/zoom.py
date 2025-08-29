import cv2
import torch
import numpy as np
from torchvision.transforms import Compose

RESOLUTION = (640, 480)
REPOSITORY = "intel-isl/MiDaS"

def load_midas_model():
    model = torch.hub.load(REPOSITORY, "MiDaS_small", skip_validation=True)
    model.eval()
    transform = torch.hub.load(REPOSITORY, "transforms").small_transform
    return model, transform

def estimate_depth(image: np.ndarray, model, transform: Compose):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb)
    print(f"input_tensor shape: {input_tensor.shape}") 
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# zoom/parallax
def generate_zoom_frames(image: np.ndarray, depth_map: np.ndarray, num_frames: int = 60, zoom_strength: float = 0.1) -> list:
    h, w = image.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = (i / (num_frames - 1)) * zoom_strength
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        shift_x = (map_x - w / 2) * depth_map * alpha
        shift_y = (map_y - h / 2) * depth_map * alpha
        new_map_x = np.clip((map_x - shift_x).astype(np.float32), 0, w - 1)
        new_map_y = np.clip((map_y - shift_y).astype(np.float32), 0, h - 1)
        warped = cv2.remap(image, new_map_x, new_map_y, interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.resize(warped, RESOLUTION))
    return frames

def save_video(frames: list, path: str, fps: float = 30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, RESOLUTION)
    for f in frames:
        out.write(f)
    out.release()
