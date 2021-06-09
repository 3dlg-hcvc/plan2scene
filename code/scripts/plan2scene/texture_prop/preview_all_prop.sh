#!/bin/bash
# This script is used to preview propagated textures by a specified checkpoint.
# Run this script from the project root directory.
# Usage:
#  CUDA_VISIBLE_DEVICES=0 /bin/bash ./code/scripts/preview_all_prop.sh ./tmp/preview_results/epoch_## [MODEL_TRAIN_OUTPUT_DIRECTORY]/conf/texture_prop.json [CHECKPOINT_PATH]
#
# Do not use this script to make the final inferences. All surfaces are predicted by the GNN (including observed surfaces).
# We simulate the predicted surface unobserved. Therefore, the GNN has to infer it using information from other observed surfaces.
# So, the performance is sub-par for observed surfaces. But, it should give a good idea about the capability of GNN to produce meaningful embeddings.

if [$# -ne 3]; then
    echo "Illegal number of parameters"
    exit 1
fi

echo "Output Path $1"
echo "GNN Network Config Path $2"
echo "Checkpoint Path $3"

# Make GNN prediction for each surface, assuming that surface is unobserved. We predict for both observed and unobserved surfaces (and hence we call it 'all prop').
python code/scripts/plan2scene/texture_prop/gnn_texture_prop.py $1 ./data/processed/vgg_crop_select/val/drop_0.0 val $2 $3 --val-graph-generator

# We skip the seam correction step in interest of time.

# Preview GNN predictions
python code/scripts/plan2scene/postprocessing/embed_textures.py $1/archs $1/texture_crops val
python code/scripts/plan2scene/render_house_jsons.py $1/archs --scene-json
python code/scripts/plan2scene/preview_houses.py $1/previews $1/archs ./data/input/photos val --textures-path $1/texture_crops 0.0

echo "Results are available at $1/previews/[HOUSE_KEY]/report.html"
