#!/bin/bash

VALIDATOR_ARGS=$@

# first, git pull
git pull

# next, set up environment
pip install -e .

# finally, run the validator
python -m neurons.validator $VALIDATOR_ARGS --auto_update
