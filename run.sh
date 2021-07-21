#!/bin/sh
conda activate object_tracking_v2
SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
echo $( dirname "${BASH_SOURCE[0]}" )
cd $SCRIPT_DIR
python main.py
conda deactivate