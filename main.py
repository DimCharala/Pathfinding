from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from scipy.spatial import KDTree, Delaunay
import copy
from collections import deque, defaultdict
import random
from itertools import combinations
import matplotlib.pyplot as plt
import heapq

WIDTH = 1600
HEIGHT = 900

class PathFinding(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "PathFinding", output=True, n_sliders=1)
        self.init("Room")

    def init(self, meshType):
        
        
        self.set_slider_value(slider_id=0, value=0.5, no_callback=True)
        self.num_samples = int(0.5 * 35000 + 25000)

        text = f"""
Select Room by pressing F1
Select Floor by pressing F2
Current number of samples: {self.num_samples}
Move the slider to change number of samples or to create the point cloud
Press A to toggle clustering(currently on)
Press M to find segments without point cloud
"""
        self.print(text)

        if meshType == "Room":
            self.mesh = Mesh3D("RoomMesh.ply", color=Color.DARKRED)
            self.meshType = meshType

            # Custom rotation around x-axis by -90 degrees
            R_x = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])

            # Custom rotation around y-axis by -90 degrees
            R_y = np.array([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]])
        
        elif meshType == "Floor":
            self.mesh = Mesh3D("FloorMesh_v2.ply", color=Color.DARKRED)
            self.meshType = meshType

            # Identity matrix (no rotation)
            R_x = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

            # Identity matrix (no rotation)
            R_y = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
            
            
        vertices = np.asarray(self.mesh.vertices)
        centroid = np.mean(vertices, axis=0)
        vertices -= centroid

        # Scaling the mesh to fit within (-1, -1, -1) to (1, 1, 1)
        max_extent = np.max(np.abs(vertices))
        vertices /= max_extent

        vertices = np.dot(vertices, R_x.T)
        vertices = np.dot(vertices, R_y.T)

        self.mesh.vertices = vertices

        self.pcd = None
        self.clusters = None
        self.rest = None
        
        
        self.showCluster = True
        self.previousClusters = 0
        self.selected_vertex = None
        self.count = 1
        self.meshSegmentExists = False
        self.grid = None
        self.grid_size = 0.05
        self.lenPath = 1

        self.addShape(self.mesh, "mesh")

    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            self.num_samples = int(value * 35000 + 25000)
            self.print(f"Slider changed: {self.num_samples} samples")
            self.creating_point_cloud()

    def creating_point_cloud(self):
        
        self.pcd = self.sampling_for_pcd(self.mesh)
        self.print("Point Cloud Created")
        self.pcd = self.process_point_cloud(self.pcd)
        
        if self.meshType == "Room":
            segments, self.rest, plane_params = self.segment_planes(self.pcd,pt_to_plane_dist=0.02,max_plane_idx= 6,max_iterations= 1000)
        elif self.meshType == "Floor":
            segments, self.rest, plane_params = self.segment_planes(self.pcd,pt_to_plane_dist=0.02,max_plane_idx= 2,max_iterations= 1000)
        self.print("Segmentation Complete")
        self.removeShape("mesh")
        
        for i in range(len(segments)):
            self.removeShape(f"segment{i}")
            self.addShape(segments[i], f"segment{i}")

        
        if self.meshType == "Room":
            self.clusters, self.noise = self.clustering(self.rest, eps=0.06-((self.num_samples-25000)/35000)*0.01, min_samples=15)
        elif self.meshType == "Floor":
            self.clusters, self.noise = self.clustering(self.rest, eps=0.04, min_samples=15)
        self.print("Clustering Complete")
        self.printCluster(self.clusters, self.noise)
        
        if self.meshType == "Room":
            # Detect the door after clustering
            door, doorUp, doorBase = self.detect_door(segments, plane_params)
            if door is not None:
                self.print("Door located")
                self.addShape(door, "door")
                
            self.doorLow = Point3D(((doorUp[0] + doorBase[0]) / 2, doorBase[1], doorBase[2]), color=Color.BLACK, size=2)
            self.lowPoint = self.doorLow.y
        elif self.meshType == "Floor":
            doorUp = np.array([0,0,0])
            self.doorLow = None
        
        # Identify the floor segment and find its convex hull
        self.floor_segment = self.identify_floor_segment(segments, plane_params)
        self.removeShape('floor_points')
        self.print("Floor segment found")
        if self.floor_segment is not None:
            
            hull_points = self.find_convex_hull(self.floor_segment.points)
            self.floor_hull_set = PointSet3D(hull_points, color=Color.CYAN, size=5)
            #self.addShape(self.floor_hull_set, "floor_points")
            if self.meshType == "Room":
                self.create_floor_grid(self.floor_hull_set, doorBase[1], doorUp[1])
            elif self.meshType == "Floor":
                heights = hull_points[:,1]
                minYindex = np.argmin(heights)
                doorBase = hull_points[minYindex]
                self.lowPoint = doorBase[1]
                self.create_floor_grid(self.floor_hull_set, doorBase[1], doorUp[1],0.0175)
        
        text = f"""
Grid created
Press left click + SHIFT to add start point
Press left click + CTRL to add end point
Press D to find path start->door
Press F to find path start->end
Press P to print grid map
"""
        self.print(text)
            

    def printCluster(self, clusters, noise):
        for i in range(self.previousClusters):
            self.removeShape(f"cluster{i}")
            self.removeShape("noise")

        if self.showCluster:
            self.print(f"Found {len(clusters)} clusters")
            self.previousClusters = len(clusters)
            for i in range(len(clusters)):
                self.addShape(clusters[i], f"cluster{i}")

            self.addShape(noise, "noise")

    def clean(self):

        if self.meshType=="Room":
            for i in range(6):
                
                self.removeShape(f'segment{i}')
                
        elif self.meshType=="Floor":
            for i in range(2):
                
                self.removeShape(f'segment{i}')
            
        for i in range(8):
            self.removeShape(f"pcd{i}")

        for i in range(self.previousClusters):
            self.removeShape(f"cluster{i}")
        self.removeShape("noise")   

        for i in range(self.count):
            self.removeShape(f"grid_cell_{i}")

        for i in range(self.lenPath):
            self.removeShape(f"path_arrow_{i}")

        self.removeShape("mesh")
        self.removeShape("door")
        self.removeShape("floor_points")
        self.removeShape("start")
        self.removeShape("end")
        return

    @world_space
    def on_mouse_press(self, x, y, z, button, modifiers):
        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
            if np.isinf(z):
                self.print('Need to click the grid plane')
                return
            if self.floor_segment is None:
                self.print("PCD not created yet")
                return
            
            if self.meshType == "Room":
                vertex_size = 1.5
            elif self.meshType == "Floor":
                vertex_size = 0.5

            self.selected_vertex = self.find_closest_vertex(self.floor_segment.points, (x, y, z))
            start = Point3D(self.selected_vertex, color=Color.RED, size=vertex_size)
            self.removeShape("start")
            self.addShape(start, "start")

        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_CTRL:
            if np.isinf(z):
                self.print('Need to click the grid plane')
                return
            if self.floor_segment is None:
                self.print("PCD not created yet")
                return

            if self.meshType == "Room":
                vertex_size = 1.5
                
            elif self.meshType == "Floor":
                vertex_size = 0.5
                

            self.selected_vertex2 = self.find_closest_vertex(self.floor_segment.points, (x, y, z))
            self.selected_vertex2 = Point3D(self.selected_vertex2, color=Color.GREEN, size=vertex_size)
            self.removeShape("end")
            self.addShape(self.selected_vertex2, "end")

    def find_closest_vertex(self, vertices, query):
        dist_sq = np.sum((vertices - query) ** 2, axis=1)
        return vertices[np.argmin(dist_sq)]

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.A:
            if self.pcd is None:
                self.print("There is no point cloud created yet")
            else:
                self.showCluster = not self.showCluster
                if self.showCluster:
                    self.print("Clustering is enabled")
                else:
                    self.print("Clustering is disabled")
                self.printCluster(self.clusters, self.rest)

        if symbol == Key.P:
            if self.selected_vertex is not None and (self.doorLow is not None or self.selected_vertex2 is not None):
                self.plot_grid()

        if symbol == Key.D:
            if self.selected_vertex is not None and self.doorLow is not None:
                self.construct_graph()
                start, goal = self.mark_start_goal_slots(self.selected_vertex, self.doorLow)
                path = self.find_path_astar(start, goal,self.lowPoint)
                self.plot_graph()
                #self.plot_path(path)

        if symbol == Key.F:
            if self.selected_vertex is not None and self.selected_vertex2 is not None:
                self.construct_graph()
                start, goal = self.mark_start_goal_slots(self.selected_vertex, self.selected_vertex2)
                path = self.find_path_astar(start, goal,self.lowPoint)
                #self.plot_path(path)
        
        if symbol == Key.M:
            if not self.meshSegmentExists and self.meshType=="Room":
                self.segment_planes_mesh()
            else:
                for i in range(8):
                    self.removeShape(f"pcd{i}")
                self.meshSegmentExists = False
        
        if symbol == Key.F1:
            self.clean()
            self.init("Room")

        if symbol == Key.F2:
            self.clean()
            self.init("Floor")


    def sampling_for_pcd(self, mesh) -> PointSet3D:
        
        #Vertices and triangle to numpy arrays
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        #Finding area size of each triangle
        areas = self.compute_triangle_areas(vertices, triangles)
        #Calculating the area samples needed in each triangle by area
        area_samples = self.calculate_samplesPerArea(areas)
        #Creating the sample points for each area
        sampled_points = self.create_points(vertices, triangles, area_samples)
        
        #Returning a PCD of the sampled points
        return PointSet3D(np.vstack((sampled_points, vertices)))

    def compute_triangle_areas(self, vertices, triangles) -> np.array:

        #Formula for calculating area of a triangle knowing the 2 vectors
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        #Finding the cross product of vectors V0->V1 and V0->V2
        cross_prod = np.cross(v1 - v0, v2 - v0)
        #Area is 0.5 * |cross_product|
        areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        return areas.reshape(-1, 1)

    def calculate_samplesPerArea(self, areas):

        #Calculating total Area
        totalArea = np.sum(areas)
        #Calculating samples per area
        areaSamples = areas / totalArea * (self.num_samples - areas.shape[0])
        #Rounding the results to have integer values
        areaSamples = np.round(areaSamples).astype(int)
        return areaSamples

    def create_points(self, vertices, triangles, area_samples) -> np.array:
        sampled_points = []
        #For each triangle
        for triangle, num_samples in zip(triangles, area_samples):
            if num_samples[0] > 0:
                #If the triangle needs sample points we append them in the sampled points
                sampled_points.append(self.sample_points_in_triangle(vertices, triangle, num_samples[0]))
        sampled_points = np.vstack(sampled_points)
        return sampled_points

    def sample_points_in_triangle(self, vertices, triangle, num_samples) -> np.array:
        v0, v1, v2 = vertices[triangle]
        #We create random values for each new sample
        r1 = np.random.uniform(size=num_samples)
        r2 = np.random.uniform(size=num_samples)
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        #P = u*A + v*B + w*C is the formula for a point
        #https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
        points = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
        return points

    def process_point_cloud(self, pcd: PointSet3D):
        
        pcd_np = np.asarray(pcd.points)
        
        # Remove statistical outliers
        pcd_np = self.remove_statistical_outliers(pcd_np, nn=16, std_multiplier=3)
        
        # Voxel downsampling
        voxel_size = 0.01
        pcd_np = self.voxel_downsample(pcd_np, voxel_size)

        return PointSet3D(pcd_np)

    def remove_statistical_outliers(self, points, nn, std_multiplier):
        #Creating a KDTree for finding nearest neighbors
        kdtree = KDTree(points)
        #Calculating distances of k nearest neighbors
        distances, _ = kdtree.query(points, k=nn)
        #Mean distance calculation
        mean_distances = np.mean(distances, axis=1)
        #Standard Deviation calculation
        std_distances = np.std(mean_distances)
        mean_of_distances = np.mean(mean_distances)
        #Determining threshold for outliers
        threshold = mean_of_distances + std_multiplier * std_distances
        
        #Identifying inliers
        inliers = points[mean_distances < threshold]
        
        return inliers

    def voxel_downsample(self, points, voxel_size):
        #Computing voxel coordinates for each point
        coords = np.floor(points / voxel_size).astype(int)
        #Finding unique voxels
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        #Selecting point for each unique voxel
        downsampled_points = points[unique_indices]
        
        return downsampled_points

    def segment_planes(self, pcd: PointSet3D, pt_to_plane_dist=0.02, max_plane_idx=6, max_iterations=1000):
        def fit_plane_ransac(points, threshold, max_iterations):
            best_eq = None
            best_inliers = []
            for _ in range(max_iterations):
                #Randomly sampling 3 points
                sample_indices = np.random.choice(points.shape[0], 3, replace=False)
                p1, p2, p3 = points[sample_indices]
                #Calculating plane parameters
                normal = np.cross(p2 - p1, p3 - p1)
                a, b, c = normal
                d = -np.dot(normal, p1)
                plane_eq = (a, b, c, d)
                #Calculating the distances of all points from the plane
                distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
                #All the points that are closer to the plane than the threshold are inliers
                inliers = np.where(distances < threshold)[0]
                #Storing the plane with the most inliers
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_eq = plane_eq
            return best_eq, best_inliers

        segments = []
        plane_params = []
        rest_points = copy.deepcopy(np.asarray(pcd.points))
        color_list = [Color.DARKRED, Color.BLUE, Color.DARKGREEN, Color.YELLOW, Color.DARKORANGE, Color.MAGENTA]

        for i in range(max_plane_idx):
            #Finding the plane
            plane_model, inliers = fit_plane_ransac(rest_points, pt_to_plane_dist, max_iterations)
            if plane_model is None or len(inliers) == 0:
                break
            segment_points = rest_points[inliers]
            #Removing the segment points from the rest points array
            rest_points = np.delete(rest_points, inliers, axis=0)
            segment_color = color_list[i % len(color_list)]
            #Creating a segment point cloud
            segment = PointSet3D(segment_points, color=segment_color)
            segments.append(segment)
            plane_params.append(plane_model)

        #Creating a point cloud for the rest of the points
        rest = PointSet3D(rest_points, color=Color.BLACK)
        return segments, rest, plane_params
    
    def segment_planes_mesh(self, pt_to_plane_dist=0.025, max_plane_idx=8, max_iterations=1500):
        def fit_plane_ransac(vertices, triangles, threshold, max_iterations):
            best_eq = None
            best_inlier_triangles = []

            triangle_centers = np.mean(vertices[triangles], axis=1)

            for _ in range(max_iterations):
                # Randomly sampling three triangles
                sample_indices = np.random.choice(triangles.shape[0], 3, replace=False)
                p1, p2, p3 = triangle_centers[sample_indices]

                # Calculate plane parameters
                normal = np.cross(p2 - p1, p3 - p1)
                if np.linalg.norm(normal) == 0:
                    continue
                normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
                a, b, c = normal
                d = -np.dot(normal, p1)
                plane_eq = (a, b, c, d)

                # Calculate the distances of all triangle centers from the plane
                distances = np.abs(np.dot(triangle_centers, normal) + d)
                inlier_triangles_mask = distances < threshold
                inlier_triangles = triangles[inlier_triangles_mask]

                # Further verify that all vertices of the inlier triangles fit the plane
                valid_inlier_triangles = []
                for tri in inlier_triangles:
                    tri_vertices = vertices[tri]
                    tri_distances = np.abs(np.dot(tri_vertices, normal) + d)
                    if np.all(tri_distances < threshold):
                        valid_inlier_triangles.append(tri)

                

                # Storing the plane with the most inlier triangles
                if len(valid_inlier_triangles) > len(best_inlier_triangles):
                    best_inlier_triangles = valid_inlier_triangles
                    best_eq = plane_eq

            
            #print(f"Best plane found: {best_eq} with {len(best_inlier_triangles)} inlier triangles")
            return best_eq, best_inlier_triangles

        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        color_list = [
            np.asarray([0, 0.5, 0]),
            np.asarray([0, 0, 0.5]),
            np.asarray([0, 0.5, 0.5]),
            np.asarray([1, 1, 1]),
            np.asarray([0, 1, 0.5]),
            np.asarray([0, 0.5, 1])
        ]

        for i in range(max_plane_idx):
            # Finding the plane
            plane_model, inlier_triangles = fit_plane_ransac(vertices, triangles, pt_to_plane_dist, max_iterations)
            if plane_model is None or len(inlier_triangles) == 0:
                break
            
            inlier_vertices = np.unique(inlier_triangles)
            segment_points = vertices[inlier_vertices]
            segment_colors = np.array([np.random.uniform(0,0.5),np.random.uniform(0,1),np.random.uniform(0,1)])
            pcd = PointSet3D(segment_points, color=segment_colors)
            self.addShape(pcd, f"pcd{i}")

            # Remove inlier triangles
            inlier_triangles_set = set(map(tuple, inlier_triangles))
            triangles = np.array([t for t in triangles if tuple(t) not in inlier_triangles_set])
            self.meshSegmentExists = True

    def clustering(self, rest: PointSet3D, eps=0.06, min_samples=20):
        def expand_cluster(points, labels, kdtree, neighbors, cluster_id, eps, min_samples):
            search_queue = deque(neighbors)  # Use deque for efficient queue operations
            while search_queue:
                point_idx = search_queue.popleft()  # Pop from the left end of the deque
                if labels[point_idx] == -1:
                    labels[point_idx] = cluster_id
                elif labels[point_idx] != -1:
                    continue
                point_neighbors = kdtree.query_ball_point(points[point_idx], eps)
                if len(point_neighbors) >= min_samples:
                    search_queue.extend(point_neighbors)  # Add new neighbors to the deque

        def dbscan(points, eps, min_samples):
            n_points = points.shape[0]
            kdtree = KDTree(points)
            labels = -1 * np.ones(n_points, dtype=int)
            cluster_id = 0
            for i in range(n_points):
                if labels[i] != -1:
                    continue
                neighbors = kdtree.query_ball_point(points[i], eps)
                if len(neighbors) >= min_samples:
                    labels[i] = cluster_id
                    expand_cluster(points, labels, kdtree, neighbors, cluster_id, eps, min_samples)
                    cluster_id += 1
            return labels

        points = np.array(rest.points)
        labels = dbscan(points, eps, min_samples)
        unique_labels = set(labels)
        clusters = []
        noise_points = points[labels == -1]  # Collect all noise points
        for k in unique_labels:
            if k == -1:
                continue  # Skip noise label
            cluster_points = points[labels == k]
            clusters.append(PointSet3D(cluster_points, color=(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), 1)))

        if len(noise_points) > 0:
            noise = PointSet3D(noise_points, color=Color.BLACK)  # Add noise points as a separate cluster

        return clusters, noise

    def detect_door(self, segments, plane_params):
        walls = []
        for i, (segment, plane_param) in enumerate(zip(segments, plane_params)):
            a, b, c, d = plane_param
            if abs(a) > 0.01 or abs(c) > 0.01:  # Assuming y-axis as up
                walls.append((segment, a, c))
        
        largest_gap = 0
        door_location = None
        door_orientation = None
        original_points = None

        for wall, a, c in walls:
            points = np.asarray(wall.points)
            if len(points) < 3:
                continue

            # Project points based on wall orientation
            if abs(c) > abs(a):  # Primarily xy-plane
                points_2d = points[:, :2]
                orientation = 'xy'
            else:  # Primarily yz-plane
                points_2d = points[:, [2, 1]]  # (z, y)
                orientation = 'yz'

            try:
                gap_location, gap_size = self.find_largest_empty_rectangle_grid(points_2d)
                if gap_size > largest_gap:
                    largest_gap = gap_size
                    door_location = gap_location
                    door_orientation = orientation
                    original_points = points
            except Exception as e:
                print(f"Error processing wall with orientation a={a}, c={c}: {e}")

        if door_location is not None:
            self.print(f"Door detected at location: {door_location}")
            door_location_3d = self.convert_to_3d(door_location, door_orientation, original_points)
            door = Cuboid3D(door_location_3d[0], door_location_3d[1], color=Color.BROWN, filled=True)
            door_height = door_location_3d[1]
            door_base = door_location_3d[0]
            return door, door_height, door_base
        return None, None,None

    def convert_to_3d(self, door_location_2d, orientation, original_points):
        if orientation == 'xy':
            (x1, y1), (x2, y2) = door_location_2d
            z_min = np.min(original_points[:, 2])
            z_max = np.max(original_points[:, 2])
            return np.array([[x1, y1, z_min], [x2, y2, z_max]])
        elif orientation == 'yz':
            (z1, y1), (z2, y2) = door_location_2d
            x_min = np.min(original_points[:, 0])
            x_max = np.max(original_points[:, 0])
            return np.array([[x_min, y1, z1], [x_max, y2, z2]])
        return None

    def find_largest_empty_rectangle_grid(self, points):
        # Define the grid size and the bounding box of the points
        grid_size = 0.01
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        width = int((x_max - x_min) / grid_size) + 1
        height = int((y_max - y_min) / grid_size) + 1

        # Create a grid to mark occupied cells
        grid = np.zeros((height, width), dtype=bool)
        for x, y in points:
            grid[int((y - y_min) / grid_size), int((x - x_min) / grid_size)] = True

        # Dynamic programming to find the largest empty rectangle
        max_area = 0
        best_rect = None
        dp = np.zeros_like(grid, dtype=int)

        for i in range(height):
            for j in range(width):
                if grid[i, j] == False:
                    dp[i, j] = dp[i - 1, j] + 1 if i > 0 else 1
                    width_hist = dp[i, j]
                    for k in range(j, -1, -1):
                        width_hist = min(width_hist, dp[i, k])
                        if width_hist == 0:
                            break
                        area = width_hist * (j - k + 1)
                        if area > max_area:
                            max_area = area
                            best_rect = ((k, i - width_hist + 1), (j, i))

        if best_rect:
            (x1_idx, y1_idx), (x2_idx, y2_idx) = best_rect
            x1, y1 = x_min + x1_idx * grid_size, y_min + y1_idx * grid_size
            x2, y2 = x_min + x2_idx * grid_size, y_min + y2_idx * grid_size
            return (np.array([x1, y1]), np.array([x2, y2])), max_area

        return None, 0
    

    def identify_floor_segment(self, segments, plane_params):
        horizontalSegments = []

        for i, (segment, plane_param) in enumerate(zip(segments, plane_params)):
            a, b, c, d = plane_param
            if abs(a) < 0.015 and abs(b) > 0.02 and abs(c) < 0.015:
                #print(f"{a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
                horizontalSegments.append([segment, i])
                
        floorSegment = None
        index = None
        minPlane = 10

        for i in range(len(horizontalSegments)):
            #print(horizontalSegments[i])
            points = np.asarray(horizontalSegments[i][0].points)
            if np.min(points[:, 1]) < minPlane:
                minPlane = np.min(points[:, 1])
                floorSegment = horizontalSegments[i][0]
                index = horizontalSegments[i][1]
        
        self.removeShape(f"segment{index}")
        return floorSegment

    def quickhull(self, points):
        def find_hull(points, hull_points, P1, P2):
            if len(points) == 0:
                return

            # Find the point P_max that is farthest from the line P1P2
            distances = np.cross(P2 - P1, points - P1)
            P_max = points[np.argmax(np.abs(distances))]

            # Add P_max to the hull
            hull_points.append(P_max)

            # Determine the points to the left of P1P_max and P_maxP2
            left_set1 = points[np.cross(P_max - P1, points - P1) > 0]
            left_set2 = points[np.cross(P2 - P_max, points - P_max) > 0]

            # Recursively find the hull points for the left sets
            find_hull(left_set1, hull_points, P1, P_max)
            find_hull(left_set2, hull_points, P_max, P2)

        # Convert points to numpy array for convenience
        points = np.array(points)

        # Find the points with the minimum and maximum x-coordinates
        min_point = points[np.argmin(points[:, 0])]
        max_point = points[np.argmax(points[:, 0])]

        # Add these points to the hull
        hull_points = [min_point, max_point]

        # Determine the points to the left and right of the line min_point-max_point
        left_set = points[np.cross(max_point - min_point, points - min_point) > 0]
        right_set = points[np.cross(min_point - max_point, points - max_point) > 0]

        # Recursively find the hull points for the left and right sets
        find_hull(left_set, hull_points, min_point, max_point)
        find_hull(right_set, hull_points, max_point, min_point)

        return np.array(hull_points)

    def find_convex_hull(self, points):
        points = np.asarray(points)
        points_2d = points[:, [0, 2]]  # Use x and z coordinates for 2D projection
        hull_points = self.quickhull(points_2d)
        return points[np.isin(points[:, [0, 2]], hull_points).all(axis=1)]

    
    def create_floor_grid(self, floor_hull, doorBase, doorHeight, grid_size=0.05):
        if floor_hull is None:
            self.print("Floor convex hull not computed yet.")
            return

        for i in range(self.count):
            self.removeShape(f"grid_cell_{i}")

        # Get the bounds of the floor hull
        hull_points = np.array(floor_hull.points[:, [0, 2]])  # Use x and z coordinates for 2D grid
        x_min, z_min = np.min(hull_points, axis=0)
        x_max, z_max = np.max(hull_points, axis=0)

        # Debug print statements to check the coordinate ranges
        #print(f"x_min: {x_min}, x_max: {x_max}")
        #print(f"z_min: {z_min}, z_max: {z_max}")

        # Create the grid
        x_coords = np.arange(x_min, x_max, grid_size)
        z_coords = np.arange(z_min, z_max, grid_size)

        # Calculate grid shape
        self.grid_shape = (len(x_coords), len(z_coords))
        #print(f"Grid shape: {self.grid_shape}")

        self.grid = {}
        for i, x in enumerate(x_coords):
            for j, z in enumerate(z_coords):
                center_x = x + grid_size / 2
                center_z = z + grid_size / 2
                self.grid[(i, j)] = (center_x, center_z, 0)

        # Get points below the door height
        below_door_points = []
        for cluster in self.clusters:
            points = np.array(cluster.points)
            below_door_points.append(points[points[:, 1] <= doorHeight])
        below_door_points = np.vstack(below_door_points)


        # Mark the grid cells as occupied if they contain any points from the clusters
        for (i, j), (center_x, center_z, status) in self.grid.items():
            cell_min_x = center_x - grid_size / 2
            cell_max_x = center_x + grid_size / 2
            cell_min_z = center_z - grid_size / 2
            cell_max_z = center_z + grid_size / 2

            # Check if any point falls within the current grid cell
            if np.any((below_door_points[:, 0] >= cell_min_x) & (below_door_points[:, 0] < cell_max_x) &
                      (below_door_points[:, 2] >= cell_min_z) & (below_door_points[:, 2] < cell_max_z)):
                self.grid[(i, j)] = (center_x, center_z, 1)


        self.count = 1
        for (i, j), (center_x, center_z, status) in self.grid.items():
            p1 = Point3D((center_x - grid_size / 2, doorBase, center_z - grid_size / 2))
            
            p2 = Point3D((center_x + grid_size / 2, doorBase + 0.001, center_z + grid_size / 2))
            
            color = Color.WHITE if status == 0 else Color.BLACK
            
            if self.meshType=="Floor":
                if status!=1:
                    cell = Cuboid3D(p1, p2, color=color, filled=True)
                    self.addShape(cell, f"grid_cell_{self.count}")
                    self.count += 1
            elif self.meshType=="Room":
                cell = Cuboid3D(p1, p2, color=color, filled=True)
                self.addShape(cell, f"grid_cell_{self.count}")
                self.count += 1
            
            

    def plot_grid(self):
        fig, ax = plt.subplots()

        # Plot grid cells
        for (i, j), (x, z, status) in self.grid.items():
            color = 'black' if status == 1 else 'white'
            rect = plt.Rectangle((x, z), self.grid_size, self.grid_size, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

        # Plot selected vertex
        if self.selected_vertex is not None:
            ax.scatter(self.selected_vertex[0], self.selected_vertex[2], color='red', s=100, label='Selected Start')

        # Plot doorLow point
        if self.doorLow is not None:
            ax.scatter(self.doorLow.x, self.doorLow.z, color='blue', s=100, label='Door')

        if self.selected_vertex2 is not None:
            ax.scatter(self.selected_vertex2.x, self.selected_vertex2.z, color='green', s=100, label='Selected End')

        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Grid Visualization')
        plt.show()

    def reset_previous_statuses(self):
        # Reset all cells with status 2 (start) or 3 (goal) to their original status
        for key, value in self.grid.items():
            if value[2] == 2 or value[2] == 3:
                # Reset status to 0 (unoccupied) or 1 (occupied)
                if (key[0], key[1]) in self.graph:
                    self.grid[key] = (value[0], value[1], 0)
                else:
                    self.grid[key] = (value[0], value[1], 1)

    def mark_start_goal_slots(self, start, goal):
        self.reset_previous_statuses()
        foundStart = False
        foundGoal = False
        
        goal = np.array([goal.x, goal.y, goal.z])
        
        startGrid = None
        goalGrid = None

        for (i, j), (x, z, status) in self.grid.items():
            cell_min_x = x
            cell_max_x = x + self.grid_size
            cell_min_z = z
            cell_max_z = z + self.grid_size

            if (start[0] >= cell_min_x) and (start[0] < cell_max_x) and (start[2] >= cell_min_z) and (start[2] < cell_max_z):
                startGrid = (i, j)
                self.grid[(i, j)] = (x, z, 2)
                foundStart = True
            
            if (goal[0] >= cell_min_x) and (goal[0] < cell_max_x) and (goal[2] >= cell_min_z) and (goal[2] < cell_max_z):
                goalGrid = (i, j)
                self.grid[(i, j)] = (x, z, 3)
                foundGoal = True

            if foundStart and foundGoal:
                return startGrid, goalGrid
            
        return startGrid, goalGrid

    def construct_graph(self):
        graph = defaultdict(list)

        for (i, j), (x, z, status) in self.grid.items():
            if status != 1:  # If the cell is not occupied
                neighbors = self.get_neighbors(i, j)
                for neighbor in neighbors:
                    graph[(i, j)].append(neighbor)

        self.graph = graph
        self.print("Graph constructed")

    def initialize_g_values(self):
        self.g_values = {node: 0 for node in self.graph}

        for (i, j), (x, z, status) in self.grid.items():
            if status == 1:
                neighbors = self.get_neighbors(i, j)
                for neighbor in neighbors:
                    if self.grid[neighbor][2] != 1:
                        self.g_values[neighbor] = 0.5

    def get_neighbors(self, i, j):
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in self.grid and self.grid[(ni, nj)][2] != 1:
                neighbors.append((ni, nj))
        return neighbors


    def find_path_astar(self, start, goal, lowPoint):

        def heuristic(a, b):
            # Use Euclidean distance as heuristic
            return np.linalg.norm(np.array(a) - np.array(b))
    
        self.initialize_g_values()

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.graph}
        g_score[start] = self.g_values[start]
        f_score = {node: float('inf') for node in self.graph}
        f_score[start] = heuristic(start, goal)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                self.plot_path(path,lowPoint)
                self.print("Path found")
                return path

            for neighbor in self.graph[current]:
                tentative_g_score = g_score[current] + self.g_values[neighbor] + heuristic(current, neighbor)

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        self.plot_path([],lowPoint)
        self.print("No path found")
        return []

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def plot_path(self, path, lowPoint):
        

        for i in range(self.lenPath):
            self.removeShape(f"path_arrow_{i}")

        if path:
            for i in range(len(path) - 1):
                start_node = path[i]
                end_node = path[i + 1]
                start_pos = self.grid[start_node][:2]
                end_pos = self.grid[end_node][:2]

                start_3d = (start_pos[0], lowPoint+0.02, start_pos[1])
                end_3d = (end_pos[0], lowPoint+0.02, end_pos[1])

                arrow = Arrow3D(start_3d, end_3d, color=Color.MAGENTA)
                self.addShape(arrow, f"path_arrow_{i}")
                self.lenPath = len(path)
        fig, ax = plt.subplots()

        # Get the bounds of the grid
        x_coords = [x for (_, _), (x, _, _) in self.grid.items()]
        z_coords = [z for (_, _), (_, z, _) in self.grid.items()]
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)

        # Plot grid cells
        for (i, j), (x, z, status) in self.grid.items():
            if status == 1:
                color = 'black'
            elif status == 2 or status == 3:
                color = 'red'
            else:
                color = 'white'
            rect = plt.Rectangle((x - self.grid_size / 2, z - self.grid_size / 2), self.grid_size, self.grid_size, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

        # Plot path
        if path:
            path_coords = np.array([(self.grid[node][0], self.grid[node][1]) for node in path])
            ax.plot(path_coords[:, 0], path_coords[:, 1], color='magenta', linewidth=2, label='Path')
            ax.scatter(path_coords[:, 0], path_coords[:, 1], color='magenta', s=50)

        ax.set_xlim(x_min - self.grid_size / 2, x_max + self.grid_size / 2)
        ax.set_ylim(z_min - self.grid_size / 2, z_max + self.grid_size / 2)
        ax.set_aspect('equal')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Grid Visualization with Path')
        plt.gca().invert_yaxis()  # Invert Y-axis to match the grid orientation
        plt.show()


    def plot_graph(self):
        fig, ax = plt.subplots()

        # Get the bounds of the grid
        x_coords = [x for (_, _), (x, _, _) in self.grid.items()]
        z_coords = [z for (_, _), (_, z, _) in self.grid.items()]
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)

        # Plot grid cells
        for (i, j), (x, z, status) in self.grid.items():
            if status == 1:
                color = 'black'
            elif status == 2:
                color = 'red'
            elif status == 3:
                color = 'green'
            else:
                color = 'white'
            rect = plt.Rectangle((x, z), self.grid_size, self.grid_size, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

        # Plot graph edges
        for (i, j), neighbors in self.graph.items():
            x1, z1 = self.grid[(i, j)][:2]
            for (ni, nj) in neighbors:
                x2, z2 = self.grid[(ni, nj)][:2]
                ax.plot([x1, x2], [z1, z2], color='blue')

        # # Plot selected vertex
        # if self.selected_vertex is not None:
        #     ax.scatter(self.selected_vertex[0], self.selected_vertex[2], color='red', s=100, label='Selected Vertex')

        # # Plot doorLow point
        # if self.doorLow is not None:
        #     ax.scatter(self.doorLow.x, self.doorLow.z, color='red', s=100, label='Door')

        ax.set_xlim(x_min, x_max + self.grid_size)
        ax.set_ylim(z_min, z_max + self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Grid and Graph Visualization')
        plt.show()


if __name__ == "__main__":
    app = PathFinding()
    app.mainLoop()
