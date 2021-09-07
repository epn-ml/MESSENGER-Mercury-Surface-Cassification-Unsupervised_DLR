
# 05-MESSENGER-MERCURY-SURFACE-CLASSIFICATION-UNSUPERVISED_DLR

## General description

Repository for the Science case from DLR with to understand the composition and evolution of planetary surfaces.

We aim to to extract the underlying information from this MESSENGER/MASCS infrared dataset using unsupervsed classification spectral data and combine/compare with chemical composition and surfaces ages inferred from crater counting.

## Analysis result

Expectations for the analysis result : 
  - Minimal : assessment if current remote sensing data from NASA/ MESSENGER could resolve different surface region of Mercury and ideas on the potential of upcoming ESA/BepiColombo mission for improvement.
  - Perfect : classifciation of Mercury surface region with uncertainties assessment, matching with laboratory measuremnts.

## Information about data set

Type of data (time series, pictures, movies, etc.)

  - Format of data : NASA PDS3
  - Size of data set : Several GB
  - Access to data set: Public from NASA/PDS Geosciences Node https://pds-geosciences.wustl.edu/missions/messenger/index.htm
  - input to the processing pipeline : csv/geojson

###  Directory structure
------------

The directory structure of your new project looks like this: 

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

Directory structure was created automatically with [cookiecutter](https://cookiecutter.readthedocs.io/en/latest/installation.html) and the template [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science#readme).
