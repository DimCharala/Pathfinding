# Room Segmentation, Clustering and Pathfinding
This project was created for the '3D Computer Vision and Geometry' course in the Electrical and Computer Engineering department at the University of Patras. It was completed in the Spring semester of the 2023-2024 academic year. The goal of this project was to segment and cluster 3D data of one or multiple rooms and determine paths for navigation within and between them.

> The vvrpywork library, developed by the Visualization & Virtual Reality Group at the University of Patras, is used in this work. For any use of this library, please contact the group via [this link](https://www.vvr.ece.upatras.gr/contact/).

## How to Install and Run
1. Download or clone the repository
2. Create a virtual environment using anaconda for Python 3.10.13
```
conda create --name myenv python=3.10.13
```
> You can replace myenv with any name of your choice.
3. Activate the virtual environment
```
conda activate myenv
```
4. Install the required dependencies
```
pip install -r requirements.txt
```
5. Navigate to the project directory and run the application
```
python main.py
```

## Completed Features
Below are all the fully implemented features of the project.

When the project starts, the initial screen appears as shown below:
![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/initial_screen.png)

- A vertical instruction panel is located on the left side of the screen.
- A slider at the bottom left allows you to create the point cloud.
- The main screen projects the scene.

### Changing between scenes
The user can switch between the two scenes using the F1 and F2 keys.
Pressing F1 loads the single-room scene, while pressing F2 loads the whole-floor scene, which includes multiple rooms.

### Creating a point
Adjusting the slider generates a point cloud containing between 25,000 and 60,000 points.

### Segmentation and clustering
Once the point cloud is created, the walls and floor are detected and segmented using the RANSAC algorithm. The remaining points are then clustered using the DBSCAN algorithm.

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/pointcloud_segmentation.png)

The user can toggle the visibility of the clusters by pressing the A key.
![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/pcd_noclusters.png)

### Door detection
Using the segmentation data for the walls, the door's position is determined by identifying the largest empty rectangle across all four walls. This method works reliably most of the time for the purposes of the project. The estimated location of a closed door is visually represented by a brown rectangle, as shown in previous images.

### Pathfinding
Pathfinding is performed using a grid graph of the floor and the A* algorithm. The floor segment is identified by detecting horizontal planes, and the grid is generated based on the floor’s boundaries, calculated from the convex hull of the 2D projections of the floor segment. Each grid cell is checked for obstacles, with cells marked as occupied (colored black) if any points from the point cloud of the clusters, and below the door’s height, intersect them. The start and end points, such as the door position, are included in the grid. Finally, the A* algorithm is used to find and display the optimal path within the grid.
The path will be visualized both in 3D and 2D views.

The user can choose the starting point, which is visualized as a red dot, by pressing SHIFT + Left Mouse Click and the ending point, which is visualized as a green dot, by pressing CTRL + Left Mouse Click. 

- To find the optimal path, which is visualized using purple arrows, from the starting point to the door, press the D key.

The 3D visualization:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/3d_visualisation_pointdoor.png)

The 2D visualization:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/2d_visualisation_pointdoor.png)

- To find the optimal path, which is visualized using purple arrows, between the two point, press the F key.

The 3D visualization:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/3d_visualisation_pointpoint.png)

The 2D visualization:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/2d_visualisation_pointpoint.png)

- To see the full graph grid map, press the P key.

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/grid_graph.png)

## Additional Features
The following features were developed as part of the project but were either not fully functional or only partially completed.

### Segmentation without using point cloud
Before creating the point cloud, the user can attempt to segment the walls and floors using the mesh triangles by pressing the M key. However, this approach was not very effective due to variations in triangle density, especially in areas with more detail.

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/segmentation_no_pointcloud.png)

The table was mistakenly detected as a plane. To address this, the program identifies two additional segments to distinguish it from the floor.

### Multiple rooms
By pressing F2 the scene changes to multiple room one. 

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/initial_multiplerooms.png)

This scene differs significantly from the single-room version, primarily due to the removal of wall segmentation. This decision was made because the RANSAC algorithm struggled with the high density of data in the vertical planes compared to the horizontal ones. As a result, the algorithm detects only the floor and ceiling segments, while the walls remain part of the clusters—a feature that was left incomplete due to project deadlines.

With clustering:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/segmentation_floor.png)

Without clustering:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/segmentation_floor_noclusters.png)

The grid is generated correctly, but the marked grid cells are not visualized due to memory limitations. By selecting two points on the floor, the optimal path between them can still be calculated.

3D visualization:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/3d_pathfinding_floor.png)

2D visualization:

![alt text](https://github.com/DimCharala/Pathfinding/blob/main/images/2d_pathfinding_floor.png)