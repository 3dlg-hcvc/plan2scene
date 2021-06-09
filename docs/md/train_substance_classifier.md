# Substance Classifier

The substance classifier is used by the __SUBS__ metric. We train it a mix of crops obtained from the opensurfaces dataset and stationary textures dataset.
More details are provided in the supplementary metrics section of the paper.
 
## Preparation of Open Surfaces Crops
Download the opensurfaces dataset. The download script will extract the rectified surface masks. 
From those extracted rectified surface masks, we prepare a `train_masks` directory and a `val_masks` directory having masks with following file names.
- [Train Masks](../txt/open_surfaces_shapes_train.txt)
- [Val Masks](../txt/open_surfaces_shapes_val.txt)

Then, we use the following script to extract crops from these rectified surface masks.
```bash
python scripts/plan2scene/metric_impl/substance_classifier/prepare_opensurfaces_crops.py ./data/processed/open-surfaces-crops/train PATH/TO/train_masks 
python scripts/plan2scene/metric_impl/substance_classifier/prepare_opensurfaces_crops.py ./data/processed/open-surfaces-crops/val PATH/TO/val_masks
```

## Preparation of Texture Dataset Crops
We use the following script to extract crops from the texture dataset.
```bash
python scripts/plan2scene/metric_impl/substance_classifier/prepare_texture_crops.py ./data/processed/stationary-textures-crops/train PATH/TO/TRAIN/TEXTURES 
python scripts/plan2scene/metric_impl/substance_classifier/prepare_texture_crops.py ./data/processed/stationary-textures-crops/val PATH/TO/VAL/TEXTURES
```

## Training Substance Classifier
1) Make sure the crops extracted from the texture dataset are in the `[PROJECT_ROOT]/data/processed/stationary-textures-dataset-crops/train` and `/data/processed/stationary-textures-crops/val`
   directories.
   
2) Make sure the crops extracted from opensurfaces rectified surface masks are in the `/data/processed/open-surfaces-crops/train` and `[PROJECT_ROOT]/data/processed/open-surfaces-crops/val`
   directories.
   
3) Run the following command to start training. This will continue training for 200 epochs.
   ```bash
   export PYTHONPATH=./code/src
   python ./code/scripts/plan2scene/metric_impl/substance_classifier/train.py ./trained_models/substance_classifier/default ./conf/plan2scene/substance_classifier_conf/default.json --save-model-interval 1
   ```
   Checkpoints are saved at './trained_models/substance_classifier/default/checkpoints' directory.
   
4) Preview learning curves using Tensorboard.
   ```bash
   tensorboard --logdir=./trained_models/substance_classifier/default/tensorboard
   ```
5) Choose the best checkpoint based on substance classification accuracy.
   Update `substance_classifier.checkpoint_path` field of `./conf/plan2scene/metric.json` to point to the best checkpoint.
   Update `substance_classifier.conf_path` field of the same file to `./trained_models/substance_classifier/default/conf/substance_classifier_conf.json`
