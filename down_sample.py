import glob
import os

import open3d as o3d
import torch
from UtilsNetwork import farthest_point_sample, index_points
from tqdm import tqdm

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for file in tqdm(glob.glob(os.path.join(base_dir, 'data/train_data', '*.ply'))):
        source = o3d.io.read_point_cloud(file)
        # source.paint_uniform_color([1, 0.706, 0])
        target = torch.tensor(source.points).unsqueeze(0).float()
        idx = farthest_point_sample(target, 512)
        target = index_points(target, idx)
        target = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target.squeeze(0)))
        o3d.io.write_point_cloud(file, target)
        # target.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([target])
    for file in tqdm(glob.glob(os.path.join(base_dir, 'data/test_data', '*.ply'))):
        source = o3d.io.read_point_cloud(file)
        # source.paint_uniform_color([1, 0.706, 0])
        target = torch.tensor(source.points).unsqueeze(0).float()
        idx = farthest_point_sample(target, 512)
        target = index_points(target, idx)
        target = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target.squeeze(0)))
        o3d.io.write_point_cloud(file, target)

    for file in tqdm(glob.glob(os.path.join(base_dir, 'data/eval_data', '*.ply'))):
        source = o3d.io.read_point_cloud(file)
        # source.paint_uniform_color([1, 0.706, 0])
        target = torch.tensor(source.points).unsqueeze(0).float()
        idx = farthest_point_sample(target, 512)
        target = index_points(target, idx)
        target = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target.squeeze(0)))
        o3d.io.write_point_cloud(file, target)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(base_dir, 'data', 'SpaceShuttle.stl')
    mesh = o3d.io.read_triangle_mesh(file)
    pcd = mesh.sample_points_uniformly(number_of_points=4096)

    pcd = torch.tensor(pcd.points).unsqueeze(0)
    pcd = pcd / torch.max(abs(pcd))
    idx = farthest_point_sample(pcd, 1024)
    pcd = index_points(pcd, idx)
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.squeeze(0)))
    file = os.path.join(base_dir, 'data', 'SpaceShuttle.ply')
    o3d.io.write_point_cloud(file, pcd)
