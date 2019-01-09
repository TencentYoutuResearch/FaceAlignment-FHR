#!/bin/bash
CUDA_VISIBLE_DEVICES=4 th main_300VW_fhr.lua -expID 300VW_fhr -trainBatch 3 -trainIters 600000 -nStack 4 -nEpochs 300 -validIters 1435 -dataset 300VW_train -task pose_fhr -outputRes 128 -output_nc 68 -nThreads 16
