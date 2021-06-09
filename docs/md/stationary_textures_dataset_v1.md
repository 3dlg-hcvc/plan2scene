# Stationary Textures Dataset - V1
The stationary textures dataset consist of textures obtained from various online sources.
We randomly sample 16 textures per category as the validation set. The remainder forms the train set. 
The images are arranged into two directories `train` and `val` at the path `[PROJECT_ROOT]/data/input/stationary-textures-dataset`. 

The following topics describe how we sample textures for each substance type.
## Wood Textures
We sourced wood textures from www.pexels.com and www.unsplash.com. 
Links to the photos we used are [available here](../txt/wood_texture_links.txt). 
We add the prefix "wood_" to the name of each photo file.

## Tile Textures
We download seamless textures under the categories 'small tiles' and 'plain tiles' from www.textures.com.
The ids of textures we downloaded are [available here](../txt/tile_texture_ids.txt).
Then, we rename the files by adding the prefix "smalltile_".

## Plaster Textures
We download seamless textures under the category 'concrete stucco' from www.textures.com.
The ids of textures we downloaded are [available here](../txt/plaster_texture_ids.txt).
Then, we rename the files by adding the prefix "plastered_".

## Carpet Textures
We downloaded seamless textures from the following sources:
- https://3djungle.net/textures/carpet/ (ids are [available here](../txt/carpet_3djungle_ids.txt))
- https://www.sketchuptextureclub.com/ (ids are [available here](../txt/carpet_sketchup_ids.txt))
- 'carpet' and 'plain fabric' categories from www.textures.com (ids are [available here](../txt/carpet_texturecom_ids.txt)).

We rename the files by adding the prefix "carpet_".