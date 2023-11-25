import librosa
from matplotlib import pyplot as plt
from moviepy.editor import AudioFileClip


class Audio:

    def __init__(self, audio_path, sample_rate=44100.):
        """
        Constructor method to load th waveform using librosa

        :param audio_path: relative path to the audio file (.wav file)
        :param sample_rate: the sample rate of the audio file in Hertz
        :return:
        """
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.audio, _ = librosa.load(self.audio_path, sr=self.sample_rate)
        self.duration = librosa.get_duration(y=self.audio, sr=self.sample_rate)
        self.moveipy_audio_clip = AudioFileClip(self.audio_path)

    def plot_waveform(self, figsize: tuple = (12, 4)):
        """
        Method to plot the audio waveform using librosa

        :param figsize: figure size in inches
        :return: librosa.display.AdaptiveWaveplot object
        """
        plt.figure(figsize=figsize)
        return librosa.display.waveshow(self.audio, sr=self.sample_rate)
