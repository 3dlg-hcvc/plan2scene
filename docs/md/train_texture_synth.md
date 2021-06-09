# Training Texture Synthesis Stage
This stage uses a modified version of the [neural texture](https://github.com/henzler/neuraltexture) project by Henzler et. al.

__IMPORTANT:__ We use a separate python environment with `torch==1.4.0` and `torchvision==0.5.0` to train the texture_gen stage. We have noticed that later versions of PyTorch causes poor training of neural texture synthesis networks. This is likely due to a compatibility issue of the noise_kernel which requires further investigation. Also, note that authors of [neural texture](https://github.com/henzler/neuraltexture) project specifies PyTorch 1.4.
You can use [this requirements file](../../code/scripts/plan2scene/texture_gen/train-requirements.txt) to setup the python environment for training.

For inference, we use the same python environment as the main project. The issue only affects training.

## Training the network
1) Copy the textures dataset to `./data/input/stationary-textures-dataset/train` and `./data/input/stationary-textures-dataset/val`.
2) Start training by running the following.
    ```bash
    python code/scripts/plan2scene/texture_gen/train.py ./trained_models/texture_gen/default ./conf/plan2scene/texture_synth_conf/v2.yml
    ```
    You may use tensorboard to preview training curves & previews, and then choose the best checkpoint based on `epoch_val_texture_loss`.
    ```bash
    python -m tensorboard --logdir ./trained_models/texture_gen/default/tensorboard
    ```
    The best checkpoints are saved at `./trained_models/texture_gen/default/best_models`. Periodic checkpoints are saved at `./trained_models/texture_gen/default/checkpoints`.
3) Update 'checkpoint_path' of `./conf/plan2scene/texture_gen.json` to the path of the selected best checkpoint from the previous step. 
   Update 'texture_synth_conf' of the same file to `./trained_models/texture_gen/default/config.yml`.
