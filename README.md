# Self-supervised image segmentation using contrastive regions
Source code for my bachelor project in Machine Learning and Data Science, at the Department of Computer Science, University of Copenhagen.

Project subsequently further developed.

## Prerequisites
To use GPUs make sure that `nvidia-container-toolkit` is installed. Install with e.g.
```bash
sudo apt install nvidia-container-toolkit
```

## Build
Build image with
```bash
sudo docker build .
```

## Run
Run the docker container interactively with e.g.
```bash
sudo docker run -it -v $(pwd)/data:/home/data -v $(pwd)/logs:/home/logs --gpus all --shm-size 8G <image ID>
```

### Train
Train a model with e.g.
```bash
train --accelerator gpu --in_channels 3 --out_channels 4 --sampler EntropySampler --batch_size 10 --crop_size 300 --dataset MoNuSegDataset --image_directory data/monuseg/ --similarity_measure MeanSquaredError --max_epochs 25 --patch_size 50 --validation_dataset MoNuSegValidationDataset --directory data/monuseg_test/
```

### Evaluate
Evaluate segmentation performance of models with e.g.
```bash
evaluate --dataset MoNuSegValidationDataset --directory data/monuseg_test/ --versions 0 1 2 3
```
