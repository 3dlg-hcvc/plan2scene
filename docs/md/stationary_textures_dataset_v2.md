# Stationary Textures Dataset - V2

| [Download](https://forms.gle/mKAmnrzAm3LCK9ua6) |
| -------- |

This dataset is a substitute to the Stationary Textures Dataset V1 described in our paper.
All the textures in this dataset are from www.cc0textures.com, made available under the [Creative Commons - CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license.

This dataset consists of 221 train set textures and 64 validation set textures. The breakdown of textures is as follows.

|Split / Substance | Wood | Carpet | Plastered | Tile |
|------------------|------|--------|-----------|------|
| Train            | 86   |36      | 47        |52    |
| Validation       | 16   |16      | 16        |16    |

The file name of a texture has the following format.
```
{substance}_{original_texture_id_from_cc0}_{original_texture_format_and_size}_crop0.jpg
```

## How did we prepare the dataset?
1) We selected suitable textures for 4 substance types from www.cc0textures.com.
2) We downloaded the 1K size version of each texture if it has a resolution of at-least 1024x1024. Otherwise, we downloaded the 2K size version.
3) We obtained a 1024x1024 crop from each texture and resized it down to 512x512.
