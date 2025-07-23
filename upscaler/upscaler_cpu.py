from basicsr.archs.rrdbnet_arch import RRDBNet
from dotenv import load_dotenv
from numpy import ndarray
from realesrgan import RealESRGANer
import cv2
import os
import torch


load_dotenv()
ESRGAN_MODEL_PATH_X2 = os.getenv('ESRGAN_MODEL_PATH_X2')
ESRGAN_MODEL_PATH_X4 = os.getenv('ESRGAN_MODEL_PATH_X4')
DEFAULT_SCALE_FACTOR = 2
DONT_OVERFLOW_RAM = False


def _get_upscaler_model(scale_factor: int) -> RealESRGANer:
    if scale_factor == 2:
        model_path = ESRGAN_MODEL_PATH_X2
    elif scale_factor == 4:
        model_path = ESRGAN_MODEL_PATH_X4
    upscaler_model = RRDBNet(
        num_in_ch = 3,
        num_out_ch = 3,
        num_feat = 64,
        num_block = 23,
        num_grow_ch = 32,
        scale = scale_factor
    )
    upscaler = RealESRGANer(
        scale = scale_factor,
        model_path = model_path,
        model = upscaler_model,
        tile = 0 if not DONT_OVERFLOW_RAM else 128,
        tile_pad = 10,
        pre_pad = 0,
        half = False,
        device = torch.device('cpu')
    )
    return upscaler


def upscale_img(model: RealESRGANer, path_input: str, scale_factor: int, path_output: str | None) -> ndarray:
    if os.path.exists(path_output):
        return
    img_source = cv2.imread(path_input)
    img_upscaled, _ = model.enhance(img_source, outscale = scale_factor)
    if (not path_output == None) and os.path.exists(path_output):
        cv2.imwrite(path_output, img_upscaled)
    return img_upscaled


def run(input_path: str, scale: int = DEFAULT_SCALE_FACTOR) -> ndarray:
    model = _get_upscaler_model(scale)
    return upscale_img(model, input_path, scale)
