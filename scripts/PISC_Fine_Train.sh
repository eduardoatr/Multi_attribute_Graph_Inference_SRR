#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

###############
##   TRAIN   ##
###############
# Model Name
Name="test"
# Metrics to Save: max, mean or vote
SaveModels="mean"
# Number of Epochs
NumEpochs=1

###############
##   MODEL   ##
###############
# Model Architecture: baseline
Architecture='edge_mlp'
# Number of Classes
NumClasses=6
# Features: bodies_activity, bodies_age, bodies_clothing, bodies_gender, context_emotion, context_activity, first_glance or objects_attention
Features="bodies_activity bodies_age bodies_clothing bodies_gender context_emotion first_glance objects_attention"
# Number of Time Steps
Time=4
# Learning Rate
LearningRate=0.0001
# Weight Regularization
Weight=0.0001
# Decay Rate
Decay=0.97
# Dropout Rate
DropoutRate=0.25
# Hidden State Size
HiddenSize=128
# Loss: cross_entropy or focal
Loss='cross_entropy'
# Reduction: sum, mean or none
Reduction='mean'
# Optimizer: adam or adamw
Optimizer='adam'
# Class Balance: weights_1, weights_2 or none
Balance='weights_2'
# Node Normalization: true or false
NodeNorm='true'
# Random Seed: int or random
Seed=67
# Run Mode: eager or graph
Mode='graph'

###############
##   PATHS   ##
###############
# Path to Features
FeaturesPath="/srv/storage/datasets/eduardo/datasets/"
# Path to Save
SavePath="/srv/storage/datasets/eduardo/models/Tensorflow/AGN/"
# Path to Results
ResultsPath="/srv/storage/datasets/eduardo/results/Tensorflow/AGN/"

###############
##   CALL    ##
###############
python3 ../social_relation_recognition/train.py \
    --name $Name \
    --save $SaveModels \
    --epochs $NumEpochs \
    --arch $Architecture \
    --classes $NumClasses \
    --features $Features \
    --time $Time \
    --learning $LearningRate \
    --weight $Weight \
    --decay $Decay \
    --dropout $DropoutRate \
    --hidden $HiddenSize \
    --loss $Loss \
    --reduction $Reduction \
    --optimizer $Optimizer \
    --balance $Balance \
    --norm $NodeNorm \
    --seed $Seed \
    --mode $Mode \
    --path_features $FeaturesPath \
    --path_save $SavePath \
    --path_results $ResultsPath