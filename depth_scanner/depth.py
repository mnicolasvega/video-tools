from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY = "intel-isl/MiDaS"
MODEL = "MiDaS_small"
DEFAULT_COLOR_MAP = "inferno"


def get_midas():
    midas = torch.hub.load(REPOSITORY, MODEL)
    midas.eval()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    midas.to(device)

    midas_transforms = torch.hub.load(REPOSITORY, "transforms")
    transform = midas_transforms.small_transform
    return midas, device, transform


def to_image(depth_map: np.ndarray, color_map: str = DEFAULT_COLOR_MAP) -> Image:
    norm = (depth_map - np.min(depth_map)) / (np.ptp(depth_map) + 1e-8) # normalize image
    colormap = plt.get_cmap(color_map)
    img_colored = colormap(norm)
    img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8) # convert to uint8 (0-255) and remove alpha channel
    return Image.fromarray(img_rgb)


def run(input_path: str) -> np.ndarray:
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    midas, device, transform = get_midas()
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
    depth_map = prediction.squeeze().cpu().numpy()
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return depth_vis
