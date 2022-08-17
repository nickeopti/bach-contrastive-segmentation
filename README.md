# Efficient Self-Supervision using Patch-based Contrastive Learning for Nuclei Segmentation

Source code for framework to self-supervisedly train convolutional neural networks to segment images – or at least learn to recognise features. Primarily developed for and tested on nuclei segmentation in histopathological images.

## Develop
The framework consists of a couple of components, and for each of these multiple different implementations can be used interchangeably. This project has some "magic" behind the scenes to facilitate fast and easy development, exploration, and testing of new implementations for these components. This has made development more efficient, but requires you to follow some processes when researching new implementations.

Following these procedures automatically makes the implementations selectable on the command line. For instance, when training, the sampler to use can be specified by the `--sampler` command line option, where any available sampler can be selected by its class name, e.g., `--sampler EntropySampler`. Furthermore, any arguments to its `__init__` method can be specified on the command line as well – without any additional code. Note that this requires using _type hints_.

### Datasets
A couple of dataset implementations are provided. To use this framework for your own data, create a new [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) in a file inside the `src/data/` directory, and decorate the class with
```python
@register_dataset(DatasetType.UNLABALLED_DATASET)
class MoNuSegDataset(Dataset):
    ...
```
or `DatasetType.LABALLED_DATASET` for datasets including labels (used for e.g. validation).

Any such decorated `Dataset` class in a file inside the `src/data/` directory will automatically be selectable on the command line by its class name on the `--dataset` and `--validation_dataset` options, respectively. And any parameters to their `__init__` methods also automatically becomes settable on the command line, such as e.g. `--image_directory` and `--crop_size` for the [`MoNuSegDataset`](https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/src/data/monuseg.py#L19).

> Note that, when e.g. using both a `--dataset` and a `--validation_dataset`, any parameters to their `__init__` methods with identical names will get the same value. To bypass this issue, just name parameters distinctly; that is the reason `MoNuSegDataset` has an `image_directory` parameter while `MoNuSegValidationDataset` has a `directory` parameter. This is a nonideal situation, but nonetheless the current situation.

### Sampling
Sampling of patches is a fundamental step in this framework. To create a new sampler, simply add a new class in [`src/sampling.py`](https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/src/sampling.py) which inherits from `Sampler`. Any such classes will automatically be selectable on the command line by its class name on the `--sampler` option.

### Similarity measures
To create a new similarity measure, simply add a new class in [`src/similarity`](https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/src/similarity.py) which inherits from `SimilarityMeasure`. Any such classes will automatically be selectable on the command line by its class name on the `--similarity_measure` option.

### Networks
To add another confidence network, simply add its class to the tuple of [`AVAILABLE_CONFIDENCE_NETWORKS`](https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/src/scripts/train.py#L20). These will be selectable on the command line by their class name on the `--confidence_network` option.

To add another featuriser network, simply add its class to the tuple of [`AVAILABLE_FEATURISER_NETWORKS`](https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/src/scripts/train.py#L23). These will be selectable on the command line by their class name on the `--featuriser_network` option.


## Install
Clone the repository, and install the package with
```sh
pip install -e .
```
inside a virtual environment. The `-e` (editable) flag is optional. Do include the dot.

> The `openslide-python` package requires [`openslide-tools`](https://openslide.org/download/), installable with e.g. `apt install openslide-tools` (on Debian/Ubunutu based systems). Alternatively, comment [the line](https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/src/data/monuseg.py#L7) out, if you are not going to be using such datasets anyways.

Alternatively, and probably recommendably, use the Docker image. See instructions below.

Installation adds the `train` and `evaluate` commands, whose usage are described below.


## Train
Train models using the framework with the `train` command, specifiying options such as e.g.
```sh
train --accelerator gpu --in_channels 3 --out_channels 4 --sampler EntropySampler --batch_size 10 --crop_size 300 --dataset MoNuSegDataset --image_directory data/monuseg/ --similarity_measure MeanSquaredError --max_epochs 25 --patch_size 50 --validation_dataset MoNuSegValidationDataset --directory data/monuseg_test/
```

## Evaluate
Evaluate trained models with the `evaluate` command
```sh
evaluate --dataset MoNuSegValidationDataset --directory data/monuseg_test/ --versions 0 1 2 3
```
where `--dataset` specifies the test set, and `--versions` lists the version numbers to evaluate.

> These version numbers correspond to the numbers you see in the `logs/lightning_logs/version_X`. Just by specifying the numbers the corresponding models are automatically loaded and evaluated.


## Docker
To be able to train on NVIDIA GPUs, make sure that `nvidia-container-toolkit` is installed.

Build the image with 
```sh
sudo docker build .
```

Run the container interactively with e.g.
```sh
sudo docker run -it -v $(pwd)/data:/home/data -v $(pwd)/logs:/home/logs --gpus all --shm-size 8G <image ID>
```
where directories containing data and logs, respectively, are mounted as volumes. This eliminates the need to copy data and logs, respectively.


## Contribute
Contributions, including new datasets and novel samplers, similarity measures, and so on are welcome. Just please adhere somewhat to `flake8` style when creating PRs.
