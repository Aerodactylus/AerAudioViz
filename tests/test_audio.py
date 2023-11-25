import librosa

from aeraudioviz.audio import Audio
from aeraudioviz.audio import AudioFeatures
from aeraudioviz.audio.feature_utils import (normalise_features, fade_in_between_times, fade_out_between_times,
                                             add_sine_wave_column, add_random_noise_column, set_to_zero_between_times)


WAV_FILE = 'tests/data/a_short_audio_sample.wav'


def test_audio_plot():
    audio = Audio(WAV_FILE)
    plot = audio.plot_waveform()
    assert type(plot) is librosa.display.AdaptiveWaveplot
