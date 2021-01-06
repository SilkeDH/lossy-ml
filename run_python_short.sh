#!/bin/sh
module load devel/jupyter_ml
date
python $1 --region globe --ouput model_8
date

