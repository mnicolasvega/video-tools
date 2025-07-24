from moviepy import VideoFileClip
from PIL import Image
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import numpy as np

DEFAULT_SCENE_THRESHOLD = 30.0 # 0.0 - 100.0

def get_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def detect_scenes(video_path: str, threshhold: float = DEFAULT_SCENE_THRESHOLD) -> list:
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold = threshhold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_manager.get_scene_list()]

def get_thumbnails(video_path: str, scenes: list) -> dict:
    thumbnails = {}
    for scene in scenes:
        start, end = scene
        clip = VideoFileClip(video_path).subclipped(start, end)

        duration = end - start
        frame_time = duration / 2
        frame_s = clip.get_frame(0)
        frame_e = clip.get_frame(frame_time)
        thumbnails[get_timestamp(start)] = Image.fromarray(np.uint8(frame_s))
        thumbnails[get_timestamp(end)] = Image.fromarray(np.uint8(frame_e))
        clip.close()
    return thumbnails

def get_clips(video_path: str, scenes: list) -> dict:
    clips = {}
    for scene in scenes:
        start, end = scene
        timestamp = get_timestamp(start)
        clip = VideoFileClip(video_path) \
            .subclipped(start, end)
        clips[timestamp] = clip
    return clips


