import librosa
from moviepy import VideoFileClip

# mp4 -> mp3
def mp4_2_mp3(video_path:str):
    video = VideoFileClip(video_path)
    audio_path = video_path[:-1] + '3'
    video.audio.write_audiofile(audio_path)
    return audio_path


def get_beats(audio_path:str): # 一个节拍对应视频一帧
    y, sr = librosa.load(audio_path)  # 加载音乐
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    if __name__ == '__main__':
        print(f"Estimated BPM: {tempo}")  # numpy.ndarray,len=1
        print(f"Beat frames: {beats}")  # numpy.ndarray
        print("节拍时间（秒）:", beat_times)  # numpy.ndarray

    return tempo,beats,beat_times

if __name__ == '__main__':
    path = mp4_2_mp3('your_video.mp4')
    get_beats(path)