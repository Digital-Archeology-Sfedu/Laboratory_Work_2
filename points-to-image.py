import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_inliers(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #Minimum and Maximum bound 
    max_bound = cloud.get_max_bound()
    min_bound = cloud.get_min_bound()
    #Define Axis Aligned Box
    axisbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    outlier_cloud.paint_uniform_color([1, 0, 0])
   # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, mesh_frame, axisbox], width=1280, height=1024)

def display_outliers(cloud, ind):
    outlier_cloud = cloud.select_by_index(ind)
    inlier_cloud = cloud.select_by_index(ind, invert=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #Minimum and Maximum bound 
    max_bound = cloud.get_max_bound()
    min_bound = cloud.get_min_bound()
    #Define Axis Aligned Box
    axisbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    outlier_cloud.paint_uniform_color([1, 0, 0])
   # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([outlier_cloud, inlier_cloud, mesh_frame, axisbox], width=1280, height=1024)

def pointcloud_to_depth_map(pointcloud: np.ndarray) -> np.ndarray:
    xs = pointcloud[:, 0]
    ys = pointcloud[:, 1]
    zs = pointcloud[:, 2]
    max_depth = zs.max()/(zs.max()-zs.min())
    rs = np.sqrt(np.square(xs) + np.square(ys) + np.square(zs))
    normalized_r = rs / max_depth
    return normalized_r

def display_cloud (pcd):
    # Axis on the Frame
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    #Minimum and Maximum bound 
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()
    #Define Axis Aligned Box
    axisbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # Visualization Input Point Cloud
    width = 1280
    height = 1024
    o3d.visualization.draw_geometries([pcd, mesh_frame, axisbox], width=width, height=height)




#Read Intup Point Cloud
pcd = o3d.io.read_point_cloud("cloud_last.ply")
#Print a summary of 3D Points
print("Input Info:", pcd)
print("Step 1 - Visualize Input Point Cloud")
display_cloud(pcd)

#Saving colors of the Input Point Cloud
save_colors = o3d.utility.Vector3dVector(pcd.colors)


#DBSCAN
#Performing the DBSCAN Clustering
labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
max_label = labels.max()
#Getting the number of Classes
print(f"Point Cloud has {max_label + 1} clusters")
#Defining the color map for clusters
colors = plt.get_cmap("tab20c")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
#Changing the colors of the Input Point Cloud
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#Visualization of the colored clusters of Point Cloud
print("Step 2 - Visualize After DBSCAN")
display_cloud(pcd)

#restoring the input colors
pcd.colors = save_colors

#Picking one cluster form point cloud which contains pipe surface
colorss = np.asarray(labels)
#Selecting the class #2 (pipe surface)
ind = np.where(colorss == 2 )[0]
pcd = pcd.select_by_index(ind)
print("Step 3 - Visualize After Selecting Pipe Class")
display_cloud(pcd)


#Get Center Point Cloud
center = pcd.get_center()
#Rotate to align point cloud
R = pcd.get_rotation_matrix_from_xyz((0, -12*np.pi/180, 0))
pcd = pcd.rotate(R, center=center)
print("Step 4 - Visualize After Aligning Surface to Axis")
display_cloud(pcd)

#Get Center Point Cloud
center = pcd.get_center()
#Rotate to align point cloud
R = pcd.get_rotation_matrix_from_xyz((0,90*np.pi/180,0))
pcd = pcd.rotate(R, center=center)
print("Step 5 - Visualize After Transforming Surfece to XY Plane")
display_cloud(pcd)


#Remove Outliers
#pcd_clear, ind = pcd.remove_statistical_outlier(nb_neighbors=4000, std_ratio=0.2)
pcd_clear, ind = pcd.remove_radius_outlier(nb_points=1000, radius=0.04)
#Colorize inlier and outlier onto surface
print("Step 6 - Visualize After Removing Radius Outliers")
display_inliers(pcd, ind)

print("Step 7 - Visualize Cleared Pipe Surface")
display_cloud(pcd_clear)


#Segment the plane
plane_model, inliers = pcd_clear.segment_plane(distance_threshold=0.0025, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
#Define Plane Equation
print(f"Calcualtion of Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
print("Step 8 - Visualize Inliers After Plane Segmentation")
display_inliers(pcd_clear, inliers)
print("Step 9 - Visualize Outliers After Plane Segmentation")
display_outliers(pcd_clear, inliers)

#Computing nearest neighbors
distances = pcd_clear.compute_nearest_neighbor_distance()
#Normals estimation
pcd_clear.estimate_normals()
avg_dist = np.mean(distances)
radius = 0.5 * avg_dist
#Create triangle mesh
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_clear,o3d.utility.DoubleVector([radius, radius * 2]))
print("Step 10 - Visualize After Surface as Mesh")
display_cloud(mesh)


import pyvista as pv
my_points = np.asarray(pcd_clear.points)
mesh = pv.PolyData(my_points)
mesh['scalars'] = mesh.points[:, 2]
mesh.plot(cmap='plasma', pbr=True, metallic=1.0, roughness=1.6, zoom=0.6)

#Project to Plane
origin = mesh.center
projected = mesh.project_points_to_plane(origin=origin)

projected.plot(cmap='plasma', pbr=True, metallic=1.0, roughness=1.6, zoom=0.6)


surf = mesh.delaunay_2d()
surf.plot(show_edges=True)