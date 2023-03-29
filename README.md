
# 05-MESSENGER-MERCURY-SURFACE-CLASSIFICATION-UNSUPERVISED_DLR

## General description

Repository for the Science case from DLR with to understand the composition and evolution of planetary surfaces.

We aim to to extract the underlying information from this MESSENGER/MASCS infrared dataset using unsupervsed classification spectral data and combine/compare with chemical composition and surfaces ages inferred from crater counting.


Please cite this code if you use it with [DOI: 10.5281/zenodo.7778172 ](https://doi.org/10.5281/zenodo.7778172)

[![DOI](https://zenodo.org/badge/322645780.svg)](https://zenodo.org/badge/latestdoi/322645780)

## Quickstart

The relevant analysis is done in `notebooks/mascs_classification_geojson.ipynb`.

`mascs_classification_geojson.py` is simply the same notebook continuosly saved as python file via [jupytext](https://jupytext.readthedocs.io/en/latest/), a Jupyter add-on.

This generate the figures needed to render the `notebooks/mascs_classification_tutorial.md`.

`mascs_classification_tutorial.html` is the markdown file rendered with [pandoc](https://pandoc.org/) and is a different output as from [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

Running the notebook could be long, we cached intermediate results `models/` and figures in `reports/figures/`.

Delete all the cached data and re-run the notebook to generate everythin anew.

## Analysis result

Expectations for the analysis result : 

  - Minimal : assessment if current remote sensing data from NASA/ MESSENGER could resolve different surface region of Mercury and ideas on the potential of upcoming ESA/BepiColombo mission for improvement.
  - Perfect : classifciation of Mercury surface region with uncertainties assessment, matching with laboratory measuremnts.

## Information about data set

Type of data:

  - Format of data : NASA [PDS3 Data Standards](https://pds.nasa.gov/datastandards/pds3/)
  - Size of data set :452 GB
  - Access to data set: Public from [NASA/PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/messenger/index.htm)
  - input to the processing pipeline : csv/geojson

###  Directory structure

The directory structure was created automatically with the template [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science#readme) of [cookiecutter](https://cookiecutter.readthedocs.io/en/latest/installation.html) tool.

Basic concepts are listed here ([extended](https://github.com/drivendata/cookiecutter-data-science/blob/master/docs/docs/index.md)) :

- **Data is immutable**. *Don't ever edit your raw data*, especially not manually, and especially not in Excel. Don't overwrite your raw data. Don't save multiple versions of the raw data. Treat the data (and its format) as immutable.
- **Notebooks are for exploration and communication**. Follow a naming convention that shows the owner and the order the analysis was done in.  Refactor the good parts. Don't write code to do the same task in multiple notebooks.
- **Analysis is a directed acyclic graph (DAG)**. Often in an analysis you have long-running steps that preprocess data or train models. If these steps have been run already (and you have stored the output somewhere like the data/interim directory), you don't want to wait to rerun them every time.
- **Keep secrets and configuration out of version control**. You really don't want to leak your AWS secret key or Postgres username and password on Github
- **Be conservative in changing the default folder structure**. To keep this structure broadly applicable for many different kinds of projects, we think the best approach is to be liberal in changing the folders around for your project, but be conservative in changing the default structure for all projects.

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│       └── test       <- minimal dataset for example and test purpose
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
