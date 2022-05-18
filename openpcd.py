import open3d as o3d

pcd = o3d.io.read_point_cloud("kimera_pcd_2.pcd")
o3d.visualization.draw_geometries([pcd])