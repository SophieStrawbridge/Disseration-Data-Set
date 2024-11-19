This folder contains all the environmental files necessary to set up the MuJoCo environment for the pick-and-place operation.

## Panda Meshes
The Panda meshes define the robot's links, how they are connected, and their behaviors. These files are essential for simulating the robot's structure and functionality within the environment.

## Environment Setup
- The robot is placed on its own table, adjacent to the workspace table where objects are positioned.  
- The camera is positioned 3 meters above the workspace table, pointing straight down at the origin to capture a top-down view of the scene.

## Object Placement
Objects are placed on the workspace table. Their configuration is determined using a parsing algorithm that modifies the file:  
`/asset/panda/assets/panda_object.xml`.  
This file specifies which object to include in the simulation environment by pointing to the appropriate object model.
