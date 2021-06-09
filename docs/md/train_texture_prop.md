# Train Texture Propagation Stage
## Pre-requisites
1) It is assumed that the texture synthesis stage is already trained, or a pre-trained model is configured.
2) Infer textures for photo observed surfaces using the VGG textureness score by
   running [vgg_crop_selector.py](code/scripts/plan2scene/crop_select/vgg_crop_selector.py). The texture propagation stage is trained using these inferences.
   ```bash
   # We simulate surface unobservations using the train_graph_generator and the val_graph_generator defined in texture_prop_conf. Therefore, drop_fraction is set to 0.0 in the following scripts.
   python code/scripts/plan2scene/preprocessing/fill_room_embeddings.py ./data/processed/texture_gen/train/drop_0.0 train --drop 0.0
   python code/scripts/plan2scene/preprocessing/fill_room_embeddings.py ./data/processed/texture_gen/val/drop_0.0 val --drop 0.0

   python code/scripts/plan2scene/crop_select/vgg_crop_selector.py ./data/processed/vgg_crop_select/train/drop_0.0 ./data/processed/texture_gen/train/drop_0.0 train --drop 0.0
   python code/scripts/plan2scene/crop_select/vgg_crop_selector.py ./data/processed/vgg_crop_select/val/drop_0.0 ./data/processed/texture_gen/val/drop_0.0 val --drop 0.0
   ```

## Training Instructions
Run the following command to start training.
```bash
python code/scripts/plan2scene/texture_prop/train.py ./trained_models/texture_prop/default
```
You may use tensorboard to preview training curves.
```bash
tensorboard --logdir ./trained_models/texture_prop/default/tensorboard
```
Checkpoints are saved at `./trained_models/texture_prop/default/checkpoints`.   

Preview results of every 50th epoch using [this script](../../code/scripts/plan2scene/texture_prop/preview_nth_epoch_all_prop.py) as shown below. 
```bash
# Preview every 50th epoch
CUDA_VISIBLE_DEVICES=0 python ./code/scripts/plan2scene/texture_prop/preview_nth_epoch_all_prop.py ./trained_models/texture_prop/default/preview_results 50 ./trained_models/texture_prop/default
# Open ./trained_models/texture_prop/default/preview_results/preview.html.
```
We usually preview every 50th epoch and choose the best epoch. Early epochs tend to look washed out with pink color shifts. 
You may also use `epoch_val_color`, `epoch_val_subs` and `epoch_val_freq` plots in tensorboard to help your decision.

For the figures reported in the paper, we used the epoch 250.
