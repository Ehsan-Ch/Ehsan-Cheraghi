# Data and attribution

Hadi Fanaee-T (2013), **Bike Sharing**, UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5W894

- Official page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Original system: Capital Bikeshare; hourly observations from 2011–2012.
- Original archive SHA-256: `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`.

The original archive is downloaded separately and not committed. Derived public
reports retain observed rental totals and timestamps for audit, plus new model
predictions. Transformations include hourly reindexing, lag/calendar feature
engineering, chronological splits and fitted median imputation. These are
changes made by this project, not claims by the original dataset creator.

The archive has 17,379 observed hourly rows. A regular grid from its first to
last timestamp has 17,544 hours, including 165 absent observations. Absent hours
are not silently converted to zero demand; their targets remain unscored.
The data contain no station identifiers, bicycle availability, price or
capacity information for causal operational decision-making.

`casual` and `registered` sum to `cnt`; they are excluded from all features.
Current-hour weather is also excluded. Only weather from the previous clock
hour is available to a forecast. Calendar flags are treated as known in advance.
Naive timestamps and unspecified release latency are acknowledged limitations.

Method background (independent implementation, not copied code):

- scikit-learn, [Lagged features for time-series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
- Angelopoulos and Bates, [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).

The chronological residual-calibration procedure here does not satisfy an
established exchangeability guarantee; its interval coverage is reported
empirically, including any failure to attain the nominal level.
