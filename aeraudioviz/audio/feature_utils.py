from typing import Optional

import numpy as np
import pandas as pd


def normalise_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def add_random_noise_column(df: pd.DataFrame, column_name='Random') -> pd.DataFrame:
    df[column_name] = np.random.rand(len(df))
    return df


def add_sine_wave_column(
        df: pd.DataFrame,
        frequency_Hz: float = 1.,
        column_name: Optional[str] = None,
        normalise: bool = True
) -> pd.DataFrame:
    """

    :param df: pandas.DataFrame with a TimeDelta index
    :param frequency_Hz: the frequency of the LFO to be added in Hertz
    :param column_name: The name of the column to add the LFO data in. If None then the column name will be of the
    form: 'LFO 1.000Hz' based on the value of frequency_Hz
    :param normalise: Boolean controlling whether the LFO data is normalised to the range (0, 1). Normalisation is
    performed when True
    :return: pandas.DataFrame with new column added containing LFO data
    """
    if column_name is None:
        column_name = f'LFO {frequency_Hz:.3f}Hz'
    wavelength = 1. / frequency_Hz
    df[column_name] = np.sin(df.index.total_seconds() % wavelength / wavelength * 2. * np.pi)
    if normalise:
        df[column_name] = normalise_features(df[column_name])
    return df


def fade_out_between_times(df, column_name, start, end, factor=1.):
    filt = (df.index.total_seconds() > start) & (df.index.total_seconds() < end)
    df.loc[filt, column_name] *= (1. - (df.loc[filt, :].index.total_seconds() - start) / (end - start)) * factor
    return df


def fade_in_between_times(df, column_name, start, end, factor=1.):
    filt = (df.index.total_seconds() >= start) & (df.index.total_seconds() < end)
    df.loc[filt, column_name] *= (df.loc[filt, :].index.total_seconds() - start) / (end - start) * factor
    return df


def set_to_zero_between_times(df, column_name, start, end):
    df.loc[(df.index.total_seconds() >= start) & (df.index.total_seconds() <= end), column_name] = 0.
    return df
