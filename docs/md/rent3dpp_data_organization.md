# Rent3D++ Dataset
The data organization is as follows.
- inputs: 
    - archs_no_objects: Contains scene.json files describing the unfurnished architecture of houses. CAD models are not placed for objects.
    - photos: Place holder directory for photos. Copy the images directory provided by Rent3D here. (Not required if you use the extracted crops provided by us.)
    - photo_assignments: Contains photoroom.csv files that describe the assignment of photos to rooms of houses.
    - object_aabbs: Contains objectaabb.json files containing labeled AABBs of objects shown in the floorplans. Provided for the test set only.
    - unobserved_photos.json: Identifies photos we unobserved for evaluation purposes.
    - data_lists: Contains house keys of train, val and test splits.
    
- processed:
    - full_archs: Architecture of houses with object CAD models placed using the approach we describe in the supplemental material section of the paper.
    - surface_crops: Extracted crops used in our paper.    

## archs_no_objects
This directory contains multiple scene.json files, each describing the architecture of a house using the scene.json format.
The filename is the house_key used to uniquely identify that house.
For details about the scene.json format, refer to [this doc](./scene_json_format.md).

The scene.json files follow the convention of considering the Y-axis as the up direction.

## object_aabbs
This directory contains multiple objectabb.json files, each containing the labeled AABBs of objects shown in the floorplan.
The file name is house_key.
The format of an objectabb.json file is as follows.
```json
"objects": [
    {
        "type": "cooking_counter", //object label
        "bound_box": {
            "p1": [ // Contains X,Z coordinates of the first point 
                7.150602191235059, 
                2.95387171314741
            ],
            "p2": [ // Contains X,Z coordinates of the second point
                7.816596812749004,
                3.6905862549800794
            ]
        }
    },
    // other objects
  ]
```

## photos
Please copy the `images` directory from the Rent 3D dataset here, if you wish to extract new surface crops.

## photo_assignments
The photo_assignments directory has multiple *.photoroom.csv files, each describing a house. 
The file name is the house_key.
Each photoroom.csv file has the following columns.
 - room_id: Room id specified for the room in the scene.json file.
 - photo: Filename of a photo assigned to the room

## full_archs
This directory has scene.json files, each describing the architecture of a house having populated with fixed objects.
The filename is the house_key used to uniquely identify that house.
You can generate full_archs from achs_no_objects and object_aabbs using the provided script.

For details about the scene.json format, refer to [this doc](./scene_json_format.md).

## surface_crops
Contains rectified surface crops extracted from Rent3D photos. These are the crops used in our paper evaluations.
You can use the provided scripts if you wish to extract new masks and crops.

## unobserved_photos.json
We evaluate Plan2Scene by simulating a random sample of photos as unobserved. For this purpose, we prepared a list of unobserved photos for each setting among 0.0 (no un-observations), 0.2, 0.4, 0.6, 0.8, and 1.0 (all photos unobserved). The setting name indicates the probability of unobserving a given photo.

`unobserved_photos.json` is a dictionary having unobserved probabilities as keys and lists of unobserved photos as values.
