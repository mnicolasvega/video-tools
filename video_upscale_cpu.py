from dotenv import load_dotenv
from pathlib import Path
import cv2
import os
import splitter.splitter
import time
import upscaler.upscaler_cpu


load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')
SCALE_FACTOR = 2
VIDEO_OUTPUT_FORMAT = "mp4"
IMAGES_OUTPUT_FORMAT = "jpg"

DIR_FRAMES_SOURCE = "source"
DIR_FRAMES_UPSCALE = "upscaled"

SKIP_FRAME_EXTRACT = False
SKIP_FRAME_UPSCALE = False
SKIP_AUDIO_EXTRACT = False


def ms_to_hms(ms: float) -> str:
    total_seconds = int(ms // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def upscale(model, image_path: str, output_path: str) -> None:
    if os.path.exists(output_path):
        frame_name = Path(output_path).stem
        print(f"skipping frame {frame_name}")
        return
    image = cv2.imread(image_path)
    upscaler.upscaler_cpu.upscale_img(model, image, SCALE_FACTOR, output_path)
    image = None

def run(video_input: str, output_dir: str) -> None:
    fps = splitter.splitter.get_fps(video_input)
    file_name = Path(video_input).stem
    audio_path = None
    if not SKIP_AUDIO_EXTRACT:
        audio_path = f"{output_dir}/{file_name}.mp3"
        if not os.path.exists(audio_path):
            splitter.splitter.extract_audio(
                video_input,
                audio_path
            )
    if not SKIP_FRAME_EXTRACT:
        splitter.splitter.split(
            video_input,
            f"{output_dir}/{DIR_FRAMES_SOURCE}",
            IMAGES_OUTPUT_FORMAT
        )
    # TODO: run over entire directory
    model = None
    if not SKIP_FRAME_UPSCALE:
        model = upscaler.upscaler_cpu.get_model(SCALE_FACTOR)
        dir_input = f"{output_dir}/{DIR_FRAMES_SOURCE}"
        dir_output = f"{output_dir}/{DIR_FRAMES_UPSCALE}"
        file_frames = sorted(os.listdir(dir_input))
        files_count = len(file_frames)
        for i, file_frame in enumerate(file_frames):
            image_path = f"{dir_input}/{file_frame}"
            output_path = f"{dir_output}/{file_frame}"
            print(f"parsing {i}/{files_count}")
            ms_start = time.time() * 1000
            upscale(model, image_path, output_path)
            ms_elapsed = (time.time() * 1000) - ms_start
            ms_estimated = ms_elapsed * (files_count - i)
            print(f"time elapsed: %.2f seg - estimated: %s" % (
                ms_elapsed / 1000,
                ms_to_hms(ms_estimated)
            ))
    splitter.splitter.join(
        f"{output_dir}/{DIR_FRAMES_UPSCALE}",
        f"{output_dir}/{file_name}.{VIDEO_OUTPUT_FORMAT}",
        fps,
        input_format = IMAGES_OUTPUT_FORMAT,
        audio_path = audio_path
    )


if __name__ == "__main__":
    file_path = Path(VIDEO_INPUT)
    file_name = file_path.stem
    output_dir = file_path.with_name(f"{file_name}")
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    for subdir in [DIR_FRAMES_SOURCE, DIR_FRAMES_UPSCALE]:
        output_subdir = Path(f"{output_dir}/{subdir}")
        output_subdir.mkdir(parents = True, exist_ok = True)
    run(VIDEO_INPUT, output_dir)
