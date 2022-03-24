#!/bin/bash

N=$1

python train.py --gpus 1 --out_channels 4 \
                --sampler EntropySampler \
                --batch_size 10 --gamma 0 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss ce --max_epochs $N

python train.py --gpus 1 --out_channels 2 \
                --sampler EntropySampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss ce --max_epochs $N

python train.py --gpus 1 --out_channels 4 \
                --sampler EntropySampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss ce --max_epochs $N
python train.py --gpus 1 --out_channels 4 \
                --sampler EntropySampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss mse --max_epochs $N
python train.py --gpus 1 --out_channels 4 \
                --sampler EntropySampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss feature  --max_epochs $N

python train.py --gpus 1 --out_channels 4 \
                --sampler TopKSampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss ce  --max_epochs $N
python train.py --gpus 1 --out_channels 4 \
                --sampler ProbabilisticSentinelSampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss ce  --max_epochs $N
