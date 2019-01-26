#!/bin.sh

find . -name "*Solutions.ipynb" -print0  | xargs -0 -I file python scripts/preprocess_notebooks.py $1 file
