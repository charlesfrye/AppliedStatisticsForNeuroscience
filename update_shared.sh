#!/bin.sh

find . -name "shared.py" -print0  | xargs -0 -I file cp "$1" file
