#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

###############
##   TEST    ##
###############
# Model Name
Name="test"
# Fusion Type: max, mean or vote
Type="mean"
# Metric Type: precision, accuracy or accuracy
Metric="precision"

###############
##   PATHS   ##
###############
# Path to Configs
ConfigsPath="/srv/storage/datasets/eduardo/results/Tensorflow/AGN/PISC/fine"
# Path to Features
FeaturesPath="/srv/storage/datasets/eduardo/datasets/"
# Path to Models
ModelsPath="/srv/storage/datasets/eduardo/models/Tensorflow/AGN/"

###############
##   CALL    ##
###############
python3 ../social_relation_recognition/test.py \
    --name $Name \
    --type $Type \
    --metric $Metric \
    --path_features $FeaturesPath \
    --path_configs $ConfigsPath \
    --path_models $ModelsPath