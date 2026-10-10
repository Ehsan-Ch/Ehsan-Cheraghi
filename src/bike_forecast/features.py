"""Clock-time lags; current measurements and target components never enter X."""
import numpy as np
import pandas as pd


def build_features(frame):
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Features require a sorted, unique hourly index')
    if len(frame) > 1 and not (frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('Regularize the hourly grid before constructing lags')
    x = pd.DataFrame(index=frame.index)
    for label, values, period in [('hour', frame.index.hour, 24),
                                   ('weekday', frame.index.dayofweek, 7),
                                   ('month', frame.index.month - 1, 12)]:
        x[label] = values
        x[label + '_sin'] = np.sin(2 * np.pi * values / period)
        x[label + '_cos'] = np.cos(2 * np.pi * values / period)
    x['holiday'] = frame.holiday
    x['workingday'] = ((frame.index.dayofweek < 5) & frame.holiday.eq(0)).astype(int)
    x['elapsed_days'] = (frame.index - pd.Timestamp('2011-01-01')).total_seconds() / 86400
    past = frame.cnt.shift(1)
    last_observed = frame.cnt.ffill().shift(1)
    for lag in (1, 2, 3, 24, 48, 168):
        x[f'count_lag_{lag}'] = frame.cnt.shift(lag)
    for window in (24, 168):
        rolling = past.rolling(window, min_periods=1)
        x[f'count_mean_{window}'] = rolling.mean()
        x[f'count_std_{window}'] = rolling.std()
        x[f'count_observed_{window}'] = rolling.count()
    for column in ['temp', 'atemp', 'hum', 'windspeed', 'weathersit']:
        x[column + '_lag_1'] = frame[column].shift(1)
    for lag in (1, 24, 168):
        x[f'naive_{lag}'] = frame.cnt.shift(lag).fillna(last_observed)
    return x


def eligible(frame, x):
    return (frame.cnt.notna() & x.naive_1.notna() &
            (frame.index >= frame.index.min() + pd.Timedelta(hours=168)))
