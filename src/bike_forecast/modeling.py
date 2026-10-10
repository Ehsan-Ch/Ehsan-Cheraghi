"""Prespecified models, nonnegative forecasts and empirical residual intervals."""
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATES = ['naive_1', 'naive_24', 'naive_168', 'ridge_1', 'ridge_100', 'hgb_15', 'hgb_31']


def fit(name, x, y, iterations=200):
    if name not in CANDIDATES:
        raise ValueError('Unknown candidate')
    model = None
    if name.startswith('ridge_'):
        model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=float(name.split('_')[1])))
    elif name.startswith('hgb_'):
        model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                              HistGradientBoostingRegressor(loss='poisson', max_iter=iterations,
                                  max_leaf_nodes=int(name.split('_')[1]), learning_rate=.1,
                                  min_samples_leaf=20, l2_regularization=1,
                                  early_stopping=False, random_state=42))
    if model is not None:
        model.fit(x, y)
    return {'name': name, 'model': model, 'columns': list(x.columns)}


def predict(bundle, x):
    if list(x.columns) != bundle['columns']:
        raise ValueError('Prediction feature schema mismatch')
    values = (x[bundle['name']].to_numpy() if bundle['model'] is None
              else bundle['model'].predict(x))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction')
    return np.maximum(values, 0)


def point_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isfinite(y + p).all():
        raise ValueError('Expected aligned finite observation and prediction vectors')
    error = y - p
    return {'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2)))}


def radius(y, p, alpha=.1):
    residual = np.abs(np.asarray(y) - np.asarray(p))
    if not 0 < alpha < 1 or not len(residual) or not np.isfinite(residual).all():
        raise ValueError('Invalid calibration data or alpha')
    rank = min(len(residual), math.ceil((len(residual) + 1) * (1 - alpha)))
    return float(np.partition(residual, rank - 1)[rank - 1])


def interval_metrics(y, lower, upper, alpha=.1):
    y, lower, upper = map(np.asarray, (y, lower, upper))
    width = upper - lower
    if np.any(width < 0):
        raise ValueError('Inverted intervals')
    interval_score = width + 2 / alpha * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    return {'coverage': float(np.mean((lower <= y) & (y <= upper))),
            'mean_width': float(np.mean(width)), 'interval_score': float(np.mean(interval_score))}
