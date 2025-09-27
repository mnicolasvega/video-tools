from dotenv import load_dotenv
import os
import time
import whisper


load_dotenv()

DIR_PATH = os.getenv('FILE_PATH')
FILE_VIDEO = os.getenv('FILE_VIDEO')
MODEL = "large-v3"
MODEL_LANGUAGE = "en"


def transcribe_srt(transcription: dict) -> str:
    subtitles = ""
    for i, segment in enumerate(transcription['segments']):
        start = segment['start']
        end = segment['end']
        text = segment['text']
        time_start = convert_to_time(start)
        time_end = convert_to_time(end)
        subtitles += f"{i+1}\n{time_start} --> {time_end}\n{text}\n\n"
    return subtitles


def transcribe_txt(transcription: dict) -> str:
    subtitles = ""
    for i, segment in enumerate(transcription['segments']):
        text = segment['text']
        subtitles += f"{text}\n"
    return subtitles


def convert_to_time(total_seconds: int) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def run(model: whisper.Whisper, path_video: str, path_srt: str, path_txt: str) -> None:
    transcript = model.transcribe(
        path_video,
        task = "translate",
        language = MODEL_LANGUAGE
    )
    print(f"-> saving: {path_srt}")
    srt_content = transcribe_srt(transcript)
    txt_content = transcribe_txt(transcript)
    with open(path_srt, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)
    with open(path_txt, "w", encoding="utf-8") as txt_file:
        txt_file.write(txt_content)
    print("-> saved")


def run_timetracking(model: whisper.Whisper, path_video: str, path_srt: str, path_txt: str) -> None:
    print(f"-> transcribing: {path_video}")
    start_time = time.time()
    run(model, path_video, path_srt, path_txt)
    end_time = time.time()
    seconds = end_time - start_time
    print(f" -> time elasped: %.2f seconds" % (seconds))


def get_model() -> whisper.Whisper:
    print(f"-> loading model: {MODEL}")
    start_time = time.time()
    model = whisper.load_model(MODEL, device="cpu")
    end_time = time.time()
    seconds = end_time - start_time
    print(f" -> time elasped: %.2f seconds" % (seconds))
    return model


def get_files(path: str) -> list:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []
    except PermissionError:
        return []


if __name__ == "__main__":
    print(f"-> loading dir: {DIR_PATH}")
    files = get_files(DIR_PATH)
    model = get_model()
    for i, file_name in enumerate(files):
        path_input = f"{DIR_PATH}/{file_name}"
        path_output_srt = f"{DIR_PATH}/{file_name}.srt"
        path_output_txt = f"{DIR_PATH}/{file_name}.txt"
        run_timetracking(model, path_input, path_output_srt, path_output_txt)
    print("-> end")
