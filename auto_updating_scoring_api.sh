#!/bin/bash

API_ARGS=$@

# first, git pull
git pull

# next, set up environment
pip install -e .

python -m nltk.downloader punkt punkt_tab

# finally, run the scoring api
python neurons/scoring_api.py $API_ARGS --auto_update
