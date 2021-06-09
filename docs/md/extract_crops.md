# [Optional] Surface Crop Extraction 
If you wish to extract new crops instead of the provided crops, follow the steps below.
   1) Extract rectified surface masks from photos using [the surface mask extractor project](https://github.com/3dlg-hcvc/plan2scene-mask-extraction).
      Note that this project uses a different Python environment, due to dependencies with prior work.

   2) Extract rectified surface crops from rectified surface masks by running [extract_surface_crops.py](https://github.com/3dlg-hcvc/plan2scene/blob/main/code/scripts/plan2scene/preprocessing/extract_surface_crops.py).
      ```bash
      python code/scripts/plan2scene/preprocessing/extract_surface_crops.py ./data/processed/rectified_crops/floor [MASK EXTRACTOR PATH]/data/output/floor/

      python code/scripts/plan2scene/preprocessing/extract_surface_crops.py ./data/processed/rectified_crops/wall [MASK EXTRACTOR PATH]/data/output/wall/

      python code/scripts/plan2scene/preprocessing/extract_surface_crops.py ./data/processed/rectified_crops/ceiling [MASK EXTRACTOR PATH]/data/output/ceiling/
      ```
