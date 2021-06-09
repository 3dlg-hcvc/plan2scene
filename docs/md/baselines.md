# Baseline Methods
## Direct Crop Baseline
This baseline uses rectified crops extracted from photos as textures.
1) Evaluate observed surfaces
   ```bash
   # Make predictions for observed surfaces
   python code/scripts/plan2scene/baseline/direct_crop_predict_observed.py ./data/processed/baselines/direct_crop/observed/test/drop_0.0 test --drop 0.0
   
   # Evaluate observed surfaces
   python code/scripts/plan2scene/test.py ./data/processed/baselines/direct_crop/observed/test/drop_0.0/texture_crops ./data/processed/gt_reference/test/texture_crops test
   ```
2) Evaluate all surfaces at 60% simulated photo unobservations.
   ```bash
   # Make predictions for observed surfaces by simulating 60% photo unobservations.
   python code/scripts/plan2scene/baseline/direct_crop_predict_observed.py ./data/processed/baselines/direct_crop/observed/test/drop_0.6 test --drop 0.6
   
   # Make predictions for unobserved surfaces.
   python code/scripts/plan2scene/baseline/direct_crop_predict_unobserved.py ./data/processed/baselines/direct_crop/all_surfaces/test/drop_0.6 ./data/processed/baselines/direct_crop/observed/test/drop_0.6/texture_crops test --drop 0.6                                                                                                                                                                                                                                          
   
   # Evaluate all surfaces
   python code/scripts/plan2scene/test.py ./data/processed/baselines/direct_crop/all_surfaces/test/drop_0.6/texture_crops ./data/processed/gt_reference/test/texture_crops test
   ```
3) Evaluate unobserved surfaces at 60% simulated photo unobservations.
   ```bash
   # We assume predictions for all surfaces are available.
   python code/scripts/plan2scene/test.py ./data/processed/baselines/direct_crop/all_surfaces/test/drop_0.6/texture_crops ./data/processed/gt_reference/test/texture_crops test --exclude-prior-predictions ./data/processed/baselines/direct_crop/observed/test/drop_0.6/texture_crops  
   ```

## Retrieve Baseline
This baseline assigns textures from the substance mapped textures (SMT) dataset.
1) Evaluate on observed surfaces.
   ```bash
   # Make predictions for observed surfaces
   python ./code/scripts/plan2scene/baseline/retrieve_predict_observed.py ./data/processed/baselines/retrieve/observed/test/drop_0.0 test --drop 0.0 [PATH_TO_SMT_DATASET]
   
   # Evaluate observed surfaces
   python code/scripts/plan2scene/test.py ./data/processed/baselines/retrieve/observed/test/drop_0.0/texture_crops ./data/processed/gt_reference/test/texture_crops test
   ```
2) Evaluate all surfaces at 60% simulated photo unobservations.
   ```bash
   # Make predictions for observed surfaces
   python ./code/scripts/plan2scene/baseline/retrieve_predict_observed.py ./data/processed/baselines/retrieve/observed/test/drop_0.6 test --drop 0.6 [PATH_TO_SMT_DATASET]
   
   # Make predictions for unobserved surfaces
   python code/scripts/plan2scene/baseline/retrieve_predict_unobserved.py ./data/processed/baselines/retrieve/all_surfaces/test/drop_0.6 ./data/processed/baselines/retrieve/observed/test/drop_0.6/texture_crops test --drop 0.6 [PATH_TO_SMT_DATASET]
   
   # Evaluate all surfaces
   python code/scripts/plan2scene/test.py ./data/processed/baselines/retrieve/all_surfaces/test/drop_0.6/texture_crops ./data/processed/gt_reference/test/texture_crops test
   ```
3) Evaluate unobserved surfaces at 60% simulated photo unobservations.
   ```bash
   # We assume predictions for all surfaces are available.
   python code/scripts/plan2scene/test.py ./data/processed/baselines/retrieve/all_surfaces/test/drop_0.6/texture_crops ./data/processed/gt_reference/test/texture_crops test --exclude-prior-predictions ./data/processed/baselines/retrieve/observed/test/drop_0.6/texture_crops  
   ```

## NaiveSynth Baseline
This baseline naively applies the unmodified neural texture synthesis approach by Henzler et al.

To use this baseline, you must first train the texture synthesis stage using the unmodified neural texture synthesis approach. 
Follow the instructions on [training the texture synthesis stage](./train_texture_synth.md) and use [naive_synth.yml](../../conf/plan2scene/texture_synth_conf/naive_synth.yml) config file instead of the `./texture_synth_conf/default.yml` file.
We suggest specifying `./trained_models/texture_synth-naivesynth/default` as the output path of train script.
Also, instead of updating the 'texture_synth_conf' and 'checkpoint_path' fields of `./conf/plan2scene/texture_gen.json` file, update the `./conf/plan2scene/texture_gen-naivesynth.json` file. 

1) Evaluate on observed surfaces.
   ```bash
   # Make predictions for observed surfaces using the mean embedding.
   python code/scripts/plan2scene/preprocessing/fill_room_embeddings.py ./data/processed/baselines/naivesynth/texture_gen/test/drop_0.0 test --drop 0.0 --texture-gen ./conf/plan2scene/texture_gen-naivesynth.json --data-paths ./conf/plan2scene/data_paths-naivesynth.json
   python code/scripts/plan2scene/baseline/naivesynth_predict_observed.py ./data/processed/baselines/naivesynth/observed/test/drop_0.0 ./data/processed/baselines/naivesynth/texture_gen/test/drop_0.0 test --drop 0.0 --texture-gen ./conf/plan2scene/texture_gen-naivesynth.json --data-paths ./conf/plan2scene/data_paths-naivesynth.json
   
   # Seam correct
   python code/scripts/plan2scene/postprocessing/seam_correct_textures.py ./data/processed/baselines/naivesynth/observed/test/drop_0.0/tileable_texture_crops ./data/processed/baselines/naivesynth/observed/test/drop_0.0/texture_crops test --drop 0.0
   
   # Evaluate observed surfaces.
   python code/scripts/plan2scene/test.py ./data/processed/baselines/naivesynth/observed/test/drop_0.0/tileable_texture_crops ./data/processed/gt_reference/test/texture_crops test
   ```
2) Evaluate all surfaces at 60% simulated photo unobservations.
   ```bash
   # Make predictions for observed surfaces using the mean embedding.
   python code/scripts/plan2scene/preprocessing/fill_room_embeddings.py ./data/processed/baselines/naivesynth/texture_gen/test/drop_0.6 test --drop 0.6 --texture-gen ./conf/plan2scene/texture_gen-naivesynth.json --data-paths ./conf/plan2scene/data_paths-naivesynth.json
   python code/scripts/plan2scene/baseline/naivesynth_predict_observed.py ./data/processed/baselines/naivesynth/observed/test/drop_0.6 ./data/processed/baselines/naivesynth/texture_gen/test/drop_0.6 test --drop 0.6 --texture-gen ./conf/plan2scene/texture_gen-naivesynth.json --data-paths ./conf/plan2scene/data_paths-naivesynth.json
   
   # Compute embeddings for the train set houses. We use these to compute RS condition mean embeddings.
   python code/scripts/plan2scene/preprocessing/fill_room_embeddings.py ./data/processed/baselines/naivesynth/texture_gen/train/drop_0.0 train --drop 0.0 --texture-gen ./conf/plan2scene/texture_gen-naivesynth.json --data-paths ./conf/plan2scene/data_paths-naivesynth.json
   
   # Make predictions for unobserved surfaces using room type and surface type conditioned mean embeddings
   python code/scripts/plan2scene/baseline/naivesynth_predict_unobserved.py ./data/processed/baselines/naivesynth/all_surfaces/test/drop_0.6 ./data/processed/baselines/naivesynth/observed/test/drop_0.6 test ./data/processed/baselines/naivesynth/texture_gen/train/drop_0.0/surface_texture_embeddings --drop 0.6 --texture-gen ./conf/plan2scene/texture_gen-naivesynth.json --data-paths ./conf/plan2scene/data_paths-naivesynth.json
   
   # Seam correct
   python code/scripts/plan2scene/postprocessing/seam_correct_textures.py ./data/processed/baselines/naivesynth/all_surfaces/test/drop_0.6/tileable_texture_crops ./data/processed/baselines/naivesynth/all_surfaces/test/drop_0.6/texture_crops test --drop 0.6
   
   # Evaluate all surfaces
   python code/scripts/plan2scene/test.py ./data/processed/baselines/naivesynth/all_surfaces/test/drop_0.6/tileable_texture_crops ./data/processed/gt_reference/test/texture_crops test
   ```
3) Evaluate unobserved surfaces at 60% simulated photo unobservations.
   ```bash
    # We assume predictions for all surfaces are available.
   python code/scripts/plan2scene/test.py ./data/processed/baselines/naivesynth/all_surfaces/test/drop_0.6/texture_crops ./data/processed/gt_reference/test/texture_crops test --exclude-prior-predictions ./data/processed/baselines/naivesynth/observed/test/drop_0.6/texture_crops
   ```