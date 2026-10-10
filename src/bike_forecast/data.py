"""Download, validate and regularize the original UCI archive."""
import hashlib
import os
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
ARCHIVE_SHA256 = 'b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(path):
    path = Path(path)
    if path.exists():
        if sha256(path) != ARCHIVE_SHA256:
            raise ValueError('Existing archive does not match the recorded data vintage')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read(5_000_001)
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Downloaded archive changed; review it before a new experiment')
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return path


def regularize(raw):
    required = {'dteday', 'hr', 'cnt', 'holiday', 'temp', 'atemp', 'hum', 'windspeed', 'weathersit'}
    if not required <= set(raw.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(raw.columns))}')
    numeric = raw[sorted(required - {'dteday'})].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Observed rows must contain finite values')
    for column in ['hr', 'cnt', 'holiday']:
        if not np.equal(numeric[column], np.floor(numeric[column])).all():
            raise ValueError(f'{column} must be integer-valued')
    if not numeric.hr.between(0, 23).all() or (numeric.cnt < 0).any():
        raise ValueError('Invalid hour or negative rental count')
    if not numeric.holiday.isin([0, 1]).all():
        raise ValueError('holiday must be binary')
    if {'casual', 'registered'} <= set(raw):
        if not np.array_equal(raw.casual + raw.registered, numeric.cnt):
            raise ValueError('Total count does not match its components')
    date = pd.to_datetime(raw.dteday, format='%Y-%m-%d', errors='raise')
    if date.isna().any():
        raise ValueError('Missing date')
    timestamps = pd.DatetimeIndex(date + pd.to_timedelta(numeric.hr, unit='h'), name='timestamp')
    if timestamps.duplicated().any():
        raise ValueError('Duplicate hourly timestamps')
    frame = numeric.copy()
    frame.index = timestamps
    frame = frame.sort_index()
    holidays = frame.groupby(frame.index.normalize()).holiday
    if (holidays.nunique() != 1).any():
        raise ValueError('Inconsistent daily holiday labels')
    holiday_map = holidays.first()
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq='h', name='timestamp')
    frame = frame.reindex(grid)
    # A known daily calendar flag is not an observation of the target hour.
    frame['holiday'] = frame.index.normalize().map(holiday_map)
    if frame.holiday.isna().any():
        raise ValueError('A whole day is missing; its holiday calendar needs explicit provision')
    return frame.drop(columns='hr')


def load_archive(path):
    if sha256(path) != ARCHIVE_SHA256:
        raise ValueError('Archive hash mismatch')
    with zipfile.ZipFile(path) as archive:
        with archive.open('hour.csv') as source:
            raw = pd.read_csv(source)
    frame = regularize(raw)
    audit = {'source_url': URL, 'archive_sha256': ARCHIVE_SHA256,
             'source_rows': len(raw), 'grid_hours': len(frame),
             'missing_hours': int(frame.cnt.isna().sum()),
             'first_hour': str(frame.index.min()), 'last_hour': str(frame.index.max())}
    return frame, audit
