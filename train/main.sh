#!/bin/bash

set -e

# set python path according to your actual environment
pythonpath='python'

${pythonpath} main.py \
            --is_train True \
            --is_test True \
            --device cuda