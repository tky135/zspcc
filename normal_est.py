import open3d as o3d

downpcd = o3d.io.read_point_cloud("00.pcd")
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  point_show_normal=True)
downpcd.orient_normals_consistent_tangent_plane(k=15)
o3d.visualization.draw_geometries([downpcd],
                                  point_show_normal=True)