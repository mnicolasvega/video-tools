import json
import os
import subprocess


DEFAULT_IMAGE_EXTENSION = "jpg"
FRAME_DIGIT_FORMAT_NUMBER = "%05d"


def split(
        video_path: str,
        dir_output: str,
        output_format: str = DEFAULT_IMAGE_EXTENSION
    ) -> None:
    os.makedirs(dir_output, exist_ok = True)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        f"{dir_output}/frame_{FRAME_DIGIT_FORMAT_NUMBER}.{output_format}"
    ]
    subprocess.run(cmd, check = True)


def join(
        frame_dir: str,
        output_path: str,
        fps: float,
        input_format: str = DEFAULT_IMAGE_EXTENSION,
        audio_path: str | None = None
    ) -> None:
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", f"{frame_dir}/frame_{FRAME_DIGIT_FORMAT_NUMBER}.{input_format}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    if not audio_path == None:
        cmd.append("-i")
        cmd.append(audio_path)
        cmd.append("-shortest") # cut at the shortest (video or audio)
    subprocess.run(cmd, check = True)


def extract_audio(
        video_path: str,
        audio_path: str
    ) -> None:
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path
    ]
    subprocess.run(cmd, check = True)


def get_fps(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-print_format", "json",
        "-show_entries", "stream=r_frame_rate",
        input_path
    ]
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = True
    )
    info = json.loads(result.stdout)
    fps_str = info["streams"][0]["r_frame_rate"]
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom
    return fps
