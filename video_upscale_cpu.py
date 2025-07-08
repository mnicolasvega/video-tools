from basicsr.archs.rrdbnet_arch import RRDBNet
from dotenv import load_dotenv
from realesrgan import RealESRGANer
from tqdm import tqdm
import cv2
import os
import subprocess
import torch



load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')
VIDEO_OUTPUT = os.getenv('VIDEO_OUTPUT')
TEMP_DIR = os.getenv('TEMP_DIR')
ESRGAN_MODEL_PATH = os.getenv('ESRGAN_MODEL_PATH')

DIR_FRAMES_SOURCE = "_frames_extracted"
DIR_FRAMES_UPSCALE = "_frames_upscale"
SCALE_FACTOR = 2 # "4" also available
VIDEO_FPS = 30 # TODO: we should use the video's FPS and not a constant



def extract_frames(video_path: str, dir_output: str) -> None:
    os.makedirs(dir_output, exist_ok = True)
    subprocess.run([
        "ffmpeg", "-i", video_path, f"{dir_output}/frame_%05d.png"
    ], check = True)



def _get_upscaler_model(model_path: str, scale_factor: int) -> RealESRGANer:
    DONT_OVERFLOW_RAM = True
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



def _upscale_img(model: RealESRGANer, path_input: str, path_output: str, scale_factor: int):
    img_source = cv2.imread(path_input)
    img_upscaled, _ = model.enhance(img_source, outscale = scale_factor)
    cv2.imwrite(path_output, img_upscaled)



def upscale_frames(dir_input: str, dir_output: str, model_path: str, scale_factor: int) -> None:
    os.makedirs(dir_output, exist_ok = True)
    model = _get_upscaler_model(model_path, scale_factor)
    FORMAT = ".png" # TODO: this should be ".png", currently a hack to reduce sample for testing
    images = sorted(
        f for f in os.listdir(dir_input) if f.endswith(FORMAT)
    ) 
    for img_name in tqdm(images, desc = "Upscaling frames"):
        path_input = os.path.join(dir_input, img_name)
        path_output = os.path.join(dir_output, img_name)
        if os.path.exists(path_output):
            print(f"  skipping {path_output}, it exists")
            continue
        _upscale_img(model, path_input, path_output, scale_factor)




def frames_to_video(frame_dir: str, output_path: str, fps: int) -> None:
    subprocess.run([
        "ffmpeg", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ], check=True)



if __name__ == "__main__":
    #extract_frames(
    #    VIDEO_INPUT,
    #    f"{TEMP_DIR}/{DIR_FRAMES_SOURCE}"
    #)
    upscale_frames(
        f"{TEMP_DIR}/{DIR_FRAMES_SOURCE}",
        f"{TEMP_DIR}/{DIR_FRAMES_UPSCALE}",
        ESRGAN_MODEL_PATH,
        SCALE_FACTOR
    )
    #frames_to_video(
    #    f"{TEMP_DIR}/{DIR_FRAMES_UPSCALE}",
    #    VIDEO_OUTPUT,
    #    1 #VIDEO_FPS
    #)
