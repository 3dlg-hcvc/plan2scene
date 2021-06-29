# Conda Environment Setup
We use a conda environment initialized as below.
- For CUDA 11.1
  
    ```bash
    # Python 3.6 and PyTorch 1.8
    conda create -n plan2scene python=3.6 -y
    conda activate plan2scene
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
    pip install -r code/requirements.txt
    
    # Install the cuda_noise package, which we have copied from the neural texture project: https://github.com/henzler/neuraltexture.
    cd code/src/plan2scene/texture_gen/custom_ops/noise_kernel
    python setup.py install
    cd ../../../../../../
    
    # Install PyTorch Geometric
    # Refer https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details.
    export CUDA=cu111 
    export TORCH=1.8.0
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache
    pip install torch-geometric --no-cache
    ```
  
- For CUDA 10.2
   ```bash
   # Python 3.6 and PyTorch 1.8
   conda create -n plan2scene python=3.6 -y
   conda activate plan2scene
   conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch -y
   pip install -r code/requirements.txt
   
   # Install the cuda_noise package, which we have copied from the neural texture project: https://github.com/henzler/neuraltexture.
   cd code/src/plan2scene/texture_gen/custom_ops/noise_kernel
   python setup.py install
   cd ../../../../../../
   
   # Install PyTorch Geometric

   # Refer https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for 
   export CUDA=cu102 details.
   export TORCH=1.8.0
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache
   pip install torch-geometric --no-cache
   ```
