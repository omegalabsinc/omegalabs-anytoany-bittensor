#!/bin/bash

API_ARGS=$@

# first, git pull
git pull

# next, set up environment
pip install -e .

# finally, run the scoring api
python neurons/scoring_api.py $API_ARGS --auto_update
