import librosa
import numpy as np
import pandas as pd

from aeraudioviz.audio import Audio
from aeraudioviz.audio.feature_utils import normalise_features


class AudioFeatures:

    def __init__(self, audio_object: Audio):
        """

        :param audio_object: object of type Audio
        :return:
        """
        self.audio = audio_object
        self.onset_envelope = self._get_onset_envelope()
        self.onset_times = self._get_onset_times()
        self.spectral_centroid = self._get_spectral_centroid()
        self.rms_values = self._get_rms_values()
        self.beat_times = self._get_beat_times()

    def _get_onset_envelope(self):
        return librosa.onset.onset_strength(
            y=self.audio.audio, sr=self.audio.sample_rate
        )

    def _get_onset_times(self):
        return librosa.times_like(self.onset_envelope, sr=self.audio.sample_rate)

    def _get_beat_times(self):
        return librosa.beat.beat_track(
            y=self.audio.audio, sr=self.audio.sample_rate, units='time'
        )[1]

    def _get_spectral_centroid(self):
        return librosa.feature.spectral_centroid(
            y=self.audio.audio, sr=self.audio.sample_rate
        )[0]

    def _get_rms_values(self):
        return librosa.feature.rms(y=self.audio.audio)[0]

    def get_feature_time_series(
            self,
            resample: bool = True,
            resample_freq: str = '41.666666L',
            spectral_centroid_rolling_mean_window: int = 48,
            beat_decay_frames: int = 5,
            beat_onset_threshold: float = .2,
            normalise: bool = True
    ) -> pd.DataFrame:
        """

        :param resample: Boolean controlling whether the time series is resampled to resample_freq
        Resample performed when true
        :param resample_freq: String to pass to pandas.DateFrame.resample method.
        Default value, 41.666666 ms, corresponds to 24 Hz (standard frame rate for video)
        :param spectral_centroid_rolling_mean_window: number of rows over which to compute spectral
        centroid rolling mean feature
        :param beat_decay_frames: number of rows over which the beat time series column decays to zero after onset of
        the detected beat
        :param beat_onset_threshold: for computing 'Beats With Decay' feature, ignore detected beats where Onset is
        below threshold
        :param normalise: Boolean to control if feature values are normalised.
        If True column-wise min-max normalisation is performed resulting in values in the range (0, 1)
        :return: pandas.DataFrame with TimeDelta index and numeric feature columns
        """
        df = pd.DataFrame(
            {'Onset': self.onset_envelope,
             'Spectral Centroid': self.spectral_centroid,
             'RMS': self.rms_values},
            index=pd.to_timedelta(self.onset_times, unit='s')
        )
        df['Spectral Centroid Rolling Mean'] = df['Spectral Centroid'].rolling(
            spectral_centroid_rolling_mean_window
        ).mean().bfill()
        if resample:
            df = df.resample(resample_freq).mean()
        df['Beats With Decay'] = self._add_beat_time_series_column(df, decay_frames=beat_decay_frames)
        df.loc[(normalise_features(df['Onset']).rolling(5).max() < beat_onset_threshold), 'Beats With Decay'] = 0.
        if normalise:
            df = normalise_features(df)
        return df

    def _add_beat_time_series_column(self, df, decay_frames: int = 5):
        t = df.index.total_seconds()
        beat_indices = np.searchsorted(t, self.beat_times, side='left') + 1
        beat_time_series = np.zeros(t.shape)
        for i in range(decay_frames):
            beat_time_series[beat_indices + i] = 1. - i / decay_frames
        return beat_time_series
