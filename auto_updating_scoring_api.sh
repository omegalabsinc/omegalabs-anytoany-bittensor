#!/bin/bash

API_ARGS=$@

# first, git pull
git pull

# next, set up environment
pip install -e .

# Install git dependencies separately (not supported in setup.py install_requires)
pip install git+https://github.com/sarulab-speech/UTMOSv2.git

python -m nltk.downloader punkt punkt_tab

# finally, run the scoring api
python -m neurons.scoring_api $API_ARGS --auto_update
