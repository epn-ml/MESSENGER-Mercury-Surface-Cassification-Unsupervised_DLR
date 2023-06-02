# Container for building the environment
FROM quay.io/condaforge/mambaforge:23.1.0-1 as conda

COPY conda-linux-64.lock .
##
RUN mamba create -n messenger-mercury-surface-cassification-unsupervised_dlr --file requirements.txt python=3
## Now add any local files from your repository.
## As an example, we add a Python package into
## the environment.
# COPY . /pipeline
# RUN conda run -p /env python -m pip install --no-deps /pkg
