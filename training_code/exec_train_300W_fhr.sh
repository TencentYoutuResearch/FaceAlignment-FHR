#!/bin/bash
CUDA_VISIBLE_DEVICES=0 th main_300W_fhr.lua -expID 300W_fhr -trainBatch 3 -trainIters 1050 -nStack 4 -nEpochs 300 -validIters 689 -dataset 300W_train_3148Set -task pose_fhr -outputRes 128 -output_nc 68 -nThreads 16
