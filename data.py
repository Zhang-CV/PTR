import os

import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def load_data(partition, num_points_src):
    # read source point clouds
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(base_dir, 'data/model', 'LRO.stl')
    mesh = o3d.io.read_triangle_mesh(file)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points_src)
    pcd = np.asarray(pcd.points)
    pcd = pcd / np.max(abs(pcd))
    pcd = np.expand_dims(pcd, axis=0)
    pcd = np.random.permutation(pcd)
    pc_sources = 0
    if partition == 'train' or 'test':
        pc_sources = np.repeat(pcd, 1024 * 4, axis=0)
    elif partition == 'eval':
        pc_sources = np.repeat(pcd, 128 * 4, axis=0)

    # read target point cloud
    files = []
    with open(os.path.join(base_dir, 'data', '%s_data' % partition, '%s_data.txt' % partition), "r") as f:
        for file in f.readlines():
            file = file.strip('\n')
            files.append(file)
    pc_means = []
    pcd_targets = []
    for file in files:
        file = os.path.join(base_dir, 'data', '%s_data' % partition, file)
        pcd = o3d.io.read_point_cloud(file)
        pcd = np.asarray(pcd.points)
        pcd = np.expand_dims(pcd, axis=0)
        pcd_mean = pcd.mean(axis=1, keepdims=False)
        pc_means.append(pcd_mean)
        pcd = pcd - pcd_mean
        pcd = np.random.permutation(pcd)
        pcd_targets.append(pcd)
    # read R and T
    pc_means = np.asarray(pc_means)
    h5_name = os.path.join(base_dir, 'data', '%s_data' % partition, 'transform.h5')
    f = h5py.File(h5_name, 'r')
    angles = f['rotation'][:].astype('float32')
    translations = f['translation'][:].astype('float32')
    f.close()
    angles = angles.T[:len(pcd_targets)]
    translations = translations.T[:len(pcd_targets)] - pc_means.squeeze(1)
    pcd_targets = np.concatenate(pcd_targets)

    return pc_sources, pcd_targets, angles, translations


def jitter_pointcloud(pointcloud, sigma=0.03, clip=0.1):
    n, c = pointcloud.shape
    # pointcloud += np.clip(sigma * np.random.randn(n, c), -1 * clip, clip)
    pointcloud += sigma * np.random.randn(n, c)
    return pointcloud


class Satellite(Dataset):
    def __init__(self, num_points_src=1024, partition='train', gaussian_noise=False):
        self.data_sources, self.data_targets, self.angles, self.translations = load_data(partition, num_points_src)
        self.partition = partition
        self.gaussian_noise = gaussian_noise

    def __getitem__(self, item):
        pts_src = self.data_sources[item]
        pts_target = self.data_targets[item]

        angle = self.angles[item]
        translation_ab = self.translations[item]

        rotation_ab = Rotation.from_euler('ZYX', angle, degrees=True)
        rotation_ab = np.asarray(rotation_ab.as_matrix()).transpose()
        if self.gaussian_noise:
            pts_target = jitter_pointcloud(pts_target)
        # return source pointcloud, target pointcloud (1, num_points, 3)
        return pts_src.astype('float32'), pts_target.astype('float32'), rotation_ab.astype('float32'), \
               translation_ab.astype('float32')

    def __len__(self):
        return len(self.data_targets)

