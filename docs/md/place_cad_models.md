# [OPTIONAL] Populate houses with CAD models of objects
We have provided scenes pre-populated with CAD models of objects.
These pre-populated scenes are available at the full_archs directory of the Rent 3D++ dataset.
You can use the smart scenes toolkit to preview these scenes.
However, if you wish to re-run this object placement step, use the commands below.

```bash
# Delete/move the full_archs directory. The following commands recreate it.
# Place CAD models for doors and windows
python code/scripts/plan2scene/place_hole_cad_models.py ./data/processed/archs_with_hole_models/test ./data/input/archs_no_objects/test

# Place CAD models for objects using the object AABBs provided by the floorplan vectorization approach.
python code/scripts/plan2scene/place_object_cad_models.py ./data/processed/full_archs/test ./data/processed/archs_with_hole_models/test ./data/input/object_aabbs/test/ 
```
