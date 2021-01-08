#!/bin/sh
module load devel/jupyter_ml
date
python $1 -r globe -o model_9
date

