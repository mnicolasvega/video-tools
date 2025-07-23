from dotenv import load_dotenv
from pathlib import Path
import os
import splitter.splitter
import upscaler.upscaler_cpu


load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')
SCALE_FACTOR = 4
VIDEO_OUTPUT_FORMAT = "mp4"
IMAGES_OUTPUT_FORMAT = "png"

DIR_FRAMES_SOURCE = "source"
DIR_FRAMES_UPSCALE = "upscaled"

SKIP_FRAME_EXTRACT = True
SKIP_FRAME_UPSCALE = True
SKIP_AUDIO_EXTRACT = True


def run(video_input: str, output_dir: str) -> None:
    fps = splitter.splitter.get_fps(video_input)
    file_name = Path(video_input).stem
    audio_path = f"{output_dir}/{file_name}.mp3"
    if not SKIP_AUDIO_EXTRACT or not os.path.exists(audio_path):
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
    # if not SKIP_FRAME_UPSCALE:
    #     upscaler.upscaler_cpu ...
    splitter.splitter.join(
        f"{output_dir}/{DIR_FRAMES_UPSCALE} x{SCALE_FACTOR}",
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
    run(VIDEO_INPUT, output_dir)
