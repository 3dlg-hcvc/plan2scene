# Scene JSON Format
In this document, we explain the scene.json format used to describe a floorplan vector. This is an extension of the scene state format used by the [smart scene toolkit](https://github.com/smartscenes/sstk/wiki). You can refer to the Rent3D++ dataset for sample scene.json files without textures. You can find a sample scene.json file with textures [here](./sample_scene.json.md).

A scene.json file contains the following information.
 - Walls are segments with two endpoints with height and depth. A wall is assigned to a room using a room id or a list of room ids.
 - Walls have holes (cutouts where windows/doors are placed) specified by a 2D bounding box on the wall surface. Optionally, the object instance id of the relevant window/door model may also be specified. 
 - Ceiling, floor, and ground surfaces are specified as polygons, each specified by an array of points. Each surface is assigned to a room using a room id.
 - Materials are specified for the two sides of the Wall (inside/outside surfaces). One material is used for all sides of the Ceiling/Floor/Ground.
 - Materials may specify textures or solid colors. When textures are used, the materials are specified in a separate section of the scene.json file, and a materialId is used to refer to it.
 - Textures are separately defined in the scene.json file, each having a texture id and a reference to an image using an image id. Materials refer to the texture using this texture id.
 - Images are separately defined in the scene.json file, each having an image id and the URL-encoded content of the image.
 - Objects are defined using a modelId (that identifies the CAD model shape), a transformation matrix, and a parent.

While the above information is sufficient to preview a scene.json file using [smart scene toolkit](https://github.com/smartscenes/sstk), we specify the following additional information for the Plan2Scene task.
 - List of rooms specifying the room ids and room types. Plan2Scene uses room-type labels for texture embedding propagation.
 - Edges of the room-door-room connectivity graph. 

 We explain the scene.json format below. 

```json
 {
    "format": "sceneState",
    "scene": {
        "up": {                                         // Up vector
            "x": 0,
            "y": 1,
            "z": 0
        },                                              // Front vector
        "front": {
            "x": 0,
            "y": 0,
            "z": 1
        },
        "unit": 1,                                      // What unit the architecture is specified in. 
                                                        // Scale to meters.
        "assetSource": [                                // Optional. Asset source containing CAD models of objects.
            "3dw"
        ],
        "assetTransforms": {                            // Optional. Tansform CAD models to match scene coordinate frame.
            "3dw": {
                "alignTo": {
                    "up": [ 0, 1, 0 ],                  // Copy scene up direction
                    "front": [ 0, 0, 1 ]                // Copy scene front direction
                },
                "scaleTo": 1,                           // Scene scale-to-metres
                "centerTo": [ 0.5, 0, 0.5 ]             // Align to center.
            }
        },
        "arch": {                                       // Describes architecture
            "id": "28025487",                           // House id
            "defaults": {                               // Optional. Specify default values for each element type.
                "Wall": {
                    "depth": 0.1,                      
                    "extraHeight": 0.035               
                },
                "Ceiling": {
                    "depth": 0.05                       
                },
                "Floor": {
                    "depth": 0.05
                }
            },
            "elements": [                               // List of architectural elements
                {                                       // Example ceiling surface
                    "id": "room_0_23ac_c",              // Unique id of the element.
                    "roomId": "room_0_23ac",            // Room id.
                    "points": [                         // Specify polygon outline of the surface.
                        [
                            [ 6.377066, 0.0, 2.103187 ],
                            [ 7.936996, 0.0, 2.103187 ],
                            // More points
                        ]
                    ],
                    "type": "Ceiling",                  // Architectural type ('Wall', 'Ceiling', 'Floor', 'Ground') 
                    "materials": [                      // Specify the material of the surface.
                        {                               // A solid color material is specified.
                            "name": "surface",
                            "diffuse": "#f6deff"
                        }
                    ],
                    "offset": [0, 2.8, 0],              // Offset added to points. 
                                                        // Here, the ceiling is lifted 2.8m from the ground. 
                    "depth": 0.05                       // Thickness of the surface.
                },
                {                                       // Example floor surface
                    "id": "room_0_23ac_f",
                    "roomId": "room_0_23ac",
                    "points": [
                        [
                            [ 6.377066, 0.0, 2.103187 ],
                            [ 7.936996, 0.0, 2.103187 ],
                            // More points
                        ]
                    ],
                    "type": "Floor",
                    "materials": [                      // Here, we refer to a material specified separately.
                        {
                            "name": "surface",
                            "materialId": "Material_room_0_23ac_f"
                        }
                    ],
                    "depth": 0.05
                },
                {                                       // Example wall
                    "roomId": "room_0_23ac",
                    "id": "room_0_23ac_wall_0",
                    "type": "Wall",
                    "points": [
                        [ 6.37706, 0.0, 2.10318 ],      // Start point of the wall
                        [ 6.37706, 0.0, 4.66693 ]       // End point of the wall
                    ],
                    "holes": [                          // List of holes in the wall. Holes make room for doors and windows.
                        {
                            "id": "hole_3f5fb0d",       // Id of hole
                            "type": "Door",             // Door or Window.
                            "box": {
                                "min": [
                                    0.12711,            // Distance from start point of the wall to the start of the hole.
                                    0.0                 // Min elevation of the hole.
                                ],
                                "max": [
                                    0.905624103585657,  // Distance from the start point of the wall to the end of the hole.
                                    2.5                 // Max elevation of the hole.
                                ]
                            }
                        },
                        // More holes
                    ],
                    "height": 2.8,                      // Height of the wall
                    "materials": [
                        {
                            "name": "surface1",         // Surface1 of wall. Interior side to assigned room.
                            "diffuse": "#f6deff"
                        },
                        {
                            "name": "surface2",         // Surface2 of wall. Exterior side to assigned room.
                            "diffuse": "#f6deff"
                        }
                    ],
                    "depth": 0.1,                       // Thickness of wall
                    "extra_height": 0.035               // This height is added to wall to ensure no gaps
                },
                // More elements
            ],
            "rdr": [                                    // Room-door-room connectivity graph edges. 
                                                        // For internal doors, specify each edge in both directions. See the example below.
                [
                    "room_0_23ac",                      // Room id of start node.
                    "hole_3f5fb0d",                     // Hole id of the door corresponding to edge.
                    "room_1_26dc"                       // Room id of the end node.
                ],
                [                                       // Same edge as before, specified in the reverse direction.
                    "room_1_26dc",                      // Room id of start node.
                    "hole_3f5fb0d",                     // Hole id of the door corresponding to edge.
                    "room_0_23ac"                       // Room id of the end node.
                ],
                // For doors facing exterior, specify only one edge as follows.
                [
                    "room_0_23ac",                     
                    "hole_65ac2d",                     
                    null                       
                ],
            ],
            "rooms": [                                  // Specify room type information of each room.
                {
                    "id": "room_1_26dc",
                    "types": [
                        "Kitchen"                       // List room types.
                    ]
                },
                // Specify details of more rooms.
            ],
            "materials": [                              // Specify materials
                {
                    "uuid": "Material_room_0_23ac_f",   // Material id
                    "map": "Texture_room_0_23ac_f"      // Id of the assigned texture.
                },
                // More materials
            ],
            "textures": [                               // Specify textures.
                {
                    "uuid": "Texture_room_0_23ac_f",    // Texture id
                    "image": "Image_room_0_23ac_f"      // Id of the assigned image.
                },
                // More textures.
            ],
            "images": [                                 // Images assigned to textures.
                {
                    "uuid": "Image_room_0_23ac_f",      // Image
                                                        // Below, you find the URL encoded image.
                    "url": "data:img/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAABC5UlEQVR4nM293XpkN64kGgC4Ut575v2vztOcB5q76S7lIoG5iABXSpZUkqy2d/XnLls/mStJEAgEAqD9n////6vMQgHITJgZbDiqgCogc8IKgDv4QwZgAgVYDIQPWASqFvJMjOGwSpQbVgG1EmZAVcFQWAW4AWMEYIZ1X4A... (truncated)"
                },
                // More images
            ]
        },
        "object":[                                      // List of CAD models
            {
                "modelId": "3dw.e1243519f6939b30601e9679255885a5",  // Shape id from shapenet
                "index": 0,                                         // Running id specified for object.
                "parentIndex": -1,                                  // No parent
                "transform": {                                      // Transformation matrix
                    "rows": 4,
                    "cols": 4,
                    "data": [
                        1.8006512615626153e-16, 0.0, 0.8109412350597608,    0.0,
                        0.0,                    1.0, 0.0,                   0.0,
                        -1.0,                   0.0, 2.220446049250313e-16, 0.0,
                        6.377066533864541,      0.0, 2.6195593625498,       1.0
                    ]
                }
            },
            // More objects
        ]
    },
    "selected": []
}
```
