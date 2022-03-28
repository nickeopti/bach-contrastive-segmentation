# Self-supervised image segmentation using contrastive regions
Source code for my bachelor project in Machine Learning and Data Science, at the Department of Computer Science, University of Copenhagen.

Currently in an almost completely undocumented state.

Train models using e.g.
```bash
python train.py --gpus 1 --out_channels 4 \
                --sampler EntropySampler \
                --batch_size 10 --gamma 0.1 --crop_size=300 \
                --dataset MoNuSegDataset --image_directory monuseg --validate_data monuseg \
                --loss ce --max_epochs $N
```

For comparing trained models, as well as visualising their training progress and test performance, create an `overview.csv` file containing information of the model training version number and model configuration description, such as 

```
816,Gamma = 0
817,2 Channels
819,Mean Squared Error
820,Featurisation
821,Top-k
822,Probabilistic
823,Cross Entropy
824,Gamma = 0
825,2 Channels
826,Cross Entropy
827,Mean Squared Error
828,Featurisation
829,Top-k
830,Probabilistic
831,Gamma = 0
832,2 Channels
833,Cross Entropy
835,Featurisation
836,Top-k
837,Probabilistic
838,Mean Squared Error
```
and run
```bash
python test_performance.py monuseg_test overview.csv
```
to select the best model within each configuration, according to the training loss, and evaluate those selected models on the test set, as well as visualising their confidence maps on the test set.

To produce the graphs of training progress, first create a concatenated csv file with all the necessary information by running
```bash
python metrics_for_vis.py overview.csv
```
and then subsequently run the `training_progress_vis.R` script.

If you wish to test this model on new datasets; fork this project, implement a corresponding dataloader in `data.py`, and add your dataloader class to the tuple of available dataloaders in https://github.com/nickeopti/bach-contrastive-segmentation/blob/main/train.py#L74. This will automatically add your dataloader (its class name) as a valid value for the `--dataset` command line parameter, as well as adding all the parameters in your dataloader's `__init__` method as command line parameters; for this to work, do add type hints to all the parameters in the `__init__` method.
