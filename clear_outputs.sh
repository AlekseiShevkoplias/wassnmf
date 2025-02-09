#!/bin/bash
for notebook in $(find . -name '*.ipynb' -not -path '*/env/*'); do
    echo "$notebook"
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$notebook"
done
