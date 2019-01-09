#!/bin/sh

# test on image benchmark: 300W
printf "Predicting coordinates ...\n"
CUDA_VISIBLE_DEVICES=1 th main_300W_fhr.lua demo
printf "Evaluating NRMSE ...\n"
matlab -nodesktop -nosplash -nojvm -r "test_300W;quit;"

## test on video benchmark: 300VW
#printf "Predicting coordinates ...\n"
#CUDA_VISIBLE_DEVICES=0 th main_300VW_fhr.lua demo
#printf "Evaluating NRMSE ...\n"
#matlab -nodesktop -nosplash -nojvm -r "test_300VW;quit;"
