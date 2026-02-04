"""
Part Segmentation Evaluation Module for Img2CAD.

This module provides functionality to evaluate part segmentation accuracy and mIoU
using a pre-trained PointNet2 model. It supports Chair, Table, and Storagefurniture categories.

The code is adapted from PointNet2 (https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
"""

import os
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree
from tqdm import tqdm
import trimesh

warnings.filterwarnings('ignore')

# Suppress trimesh logging
import logging
trimesh.util.attach_to_log(level=logging.ERROR)


# ============================================================================
# Part Segmentation Class Definitions
# ============================================================================

# ShapeNet Part Segmentation classes (for Chair, Table)
SHAPENET_SEG_CLASSES = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 
    'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 
    'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 
    'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
}

# Custom classes for Storagefurniture (7 parts)
STORAGEFURNITURE_SEG_CLASSES = {
    'Storagefurniture': [0, 1, 2, 3, 4, 5, 6]
}


def get_seg_classes(category: str) -> Dict[str, List[int]]:
    """Get segmentation class mapping for a given category."""
    if category.lower() == 'storagefurniture':
        return STORAGEFURNITURE_SEG_CLASSES
    return SHAPENET_SEG_CLASSES


def build_seg_label_to_cat(seg_classes: Dict[str, List[int]]) -> Dict[int, str]:
    """Build mapping from segment label to category name."""
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    return seg_label_to_cat


# ============================================================================
# Utility Functions
# ============================================================================

def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def find_nearest_neighbors(pc1: np.ndarray, pc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find nearest neighbor in pc2 for each point in pc1 using KDTree."""
    tree = KDTree(pc2)
    distances, indices = tree.query(pc1)
    return indices, distances


def extract_digits(string: str) -> List[str]:
    """Extract all digits from a string."""
    return re.findall(r'\d', string)


def get_3D_rot_matrix(axis: int, angle: float) -> np.ndarray:
    """Get 3D rotation matrix for given axis (0=x, 1=y, 2=z) and angle in radians."""
    if axis == 0:
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 1:
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:  # axis == 2
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])


def to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode a tensor."""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


# ============================================================================
# PointNet2 Model Components
# ============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance between each pair of points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points from a point cloud."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Ball query for local region."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, 
                     points: torch.Tensor, returnfps: bool = False):
    """Sample and group points."""
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor):
    """Sample and group all points."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet Set Abstraction layer."""
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """PointNet Set Abstraction with Multi-Scale Grouping."""
    
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """PointNet Feature Propagation layer."""
    
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2PartSeg(nn.Module):
    """PointNet2 Part Segmentation model."""
    
    def __init__(self, num_part: int = 50, normal_channel: bool = False):
        super().__init__()
        additional_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [32, 64, 128], 
            3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 
            128 + 128 + 64, [[128, 128, 256], [128, 196, 256]]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150 + additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz: torch.Tensor, cls_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
            
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


# ============================================================================
# Dataset Classes
# ============================================================================

class PartSegDataset(Dataset):
    """
    Dataset for part segmentation evaluation.
    Supports Chair, Table, and Storagefurniture using unified txt format annotations.
    All annotations are stored in data/shapenet_partseg/{category_synset}/*.txt
    """
    
    def __init__(
        self, 
        obj_dir: str,
        category: str,
        n_points: int = 2048,
        data_root: str = 'data',
    ):
        """
        Args:
            obj_dir: Directory containing predicted .obj files (or subdirs with final.obj)
            category: Category name ('Chair', 'Table', 'Storagefurniture')
            n_points: Number of points to sample
            data_root: Root directory for data files
        """
        self.obj_dir = Path(obj_dir)
        self.category = category
        self.n_points = n_points
        self.data_root = Path(data_root)
        self.normal_channel = False
        
        # Load test IDs
        splits_dir = self.data_root / 'splits'
        id_file = splits_dir / f'{category.lower()}_test_ids.txt'
        with open(id_file, 'r') as f:
            self.test_ids = [line.strip() for line in f.readlines()]
        
        # Find obj files
        self.file_mapping = self._find_obj_files()
        
        # Setup annotation data (unified for all categories)
        self._setup_annotation_data()
    
    def _find_obj_files(self) -> Dict[str, Optional[str]]:
        """Find .obj files for each test ID."""
        file_mapping = {}
        
        for test_id in self.test_ids:
            obj_path = None
            
            # Check for subdirectory with final.obj (TrAssembler output format)
            subdir_path = self.obj_dir / str(test_id) / 'final.obj'
            if subdir_path.exists():
                obj_path = str(subdir_path)
            else:
                # Check for direct .obj files matching the ID
                for f in self.obj_dir.glob('*.obj'):
                    if 'gt' not in f.name:
                        file_id = ''.join(extract_digits(f.name))
                        if file_id == str(test_id):
                            obj_path = str(f)
                            break
            
            file_mapping[str(test_id)] = obj_path
        
        return file_mapping
    
    def _setup_annotation_data(self):
        """Setup part annotation data for all categories using unified txt format."""
        # Part annotation data path (in data_root/shapenet_partseg)
        self.anno_data_dir = self.data_root / 'shapenet_partseg'
        self.synset_file = self.anno_data_dir / 'synsetoffset2category.txt'
        
        # Load category to synset mapping
        self.cat = {}
        if self.synset_file.exists():
            with open(self.synset_file, 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    if len(ls) >= 2:
                        self.cat[ls[0]] = ls[1]
        else:
            print(f"Warning: Synset file not found: {self.synset_file}")
        
        # Get synset for this category
        self.category_synset = self.cat.get(self.category, self.category.lower())
        self.category_dir = self.anno_data_dir / self.category_synset
        
        # For Chair/Table, we need anno_id to model_id mapping
        # For Storagefurniture, the annotation files are named by test_id directly
        self.use_id_mapping = self.category.lower() in ['chair', 'table']
        
        if self.use_id_mapping:
            mapping_file = self.data_root / f'anno_id2model_id_{self.category.lower()}.json'
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    self.anno_to_model = json.load(f)
            else:
                print(f"Warning: Anno mapping file not found: {mapping_file}")
                self.anno_to_model = {}
        else:
            self.anno_to_model = {}
        
        # Filter valid samples
        self.valid_ids = []
        self.anno_ids = []  # The ID used to find annotation files
        
        for test_id in self.test_ids:
            if self.use_id_mapping:
                # Chair/Table: use mapping to get ShapeNet model ID
                if str(test_id) in self.anno_to_model:
                    anno_id = self.anno_to_model[str(test_id)]
                    anno_file = self.category_dir / f'{anno_id}.txt'
                    if anno_file.exists():
                        self.valid_ids.append(str(test_id))
                        self.anno_ids.append(anno_id)
            else:
                # Storagefurniture: annotation files are named by test_id
                anno_file = self.category_dir / f'{test_id}.txt'
                if anno_file.exists():
                    self.valid_ids.append(str(test_id))
                    self.anno_ids.append(str(test_id))
        
        print(f"Found {len(self.valid_ids)} valid samples with annotations out of {len(self.test_ids)} test IDs")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        test_id = self.valid_ids[idx]
        anno_id = self.anno_ids[idx]
        
        obj_path = self.file_mapping.get(test_id)
        
        # Load and sample from mesh
        if obj_path and Path(obj_path).exists():
            try:
                mesh = trimesh.load(obj_path)
                cur_points = trimesh.sample.sample_surface_even(mesh, self.n_points)[0]
                if len(cur_points) < self.n_points:
                    add_points = mesh.sample(self.n_points - len(cur_points))
                    cur_points = np.concatenate([cur_points, add_points], axis=0)
            except Exception:
                cur_points = np.random.randn(self.n_points, 3).astype(np.float32)
        else:
            cur_points = np.random.randn(self.n_points, 3).astype(np.float32)
        
        cur_points = pc_normalize(cur_points.astype(np.float32))
        
        # For Chair/Table (ShapeNet data), apply coordinate transform
        if self.use_id_mapping:
            rot_mat = get_3D_rot_matrix(1, -np.pi / 2).astype(np.float32)
            cur_points = cur_points @ rot_mat
        
        # Load annotation from txt file (unified format for all categories)
        anno_file = self.category_dir / f'{anno_id}.txt'
        
        if anno_file.exists():
            data = np.loadtxt(str(anno_file)).astype(np.float32)
            gt_points = data[:, :3]
            seg = data[:, -1].astype(np.int32)
            
            gt_points = pc_normalize(gt_points)
            
            # Resample GT points
            choice = np.random.choice(len(seg), self.n_points, replace=True)
            gt_points = gt_points[choice]
            seg = seg[choice]
            
            # Transfer labels using nearest neighbor
            indices, _ = find_nearest_neighbors(cur_points, gt_points)
            seg = seg[indices]
        else:
            seg = np.zeros(self.n_points, dtype=np.int32)
        
        # Class label
        classes = dict(zip(self.cat.keys(), range(len(self.cat))))
        cls = np.array([classes.get(self.category, 0)]).astype(np.int32)
        
        return cur_points, cls, seg, test_id


# ============================================================================
# Evaluation Functions
# ============================================================================

def load_partseg_model(
    checkpoint_path: str,
    num_part: int = 50,
    normal_channel: bool = False,
    device: str = 'cuda'
) -> PointNet2PartSeg:
    """Load a pre-trained PointNet2 part segmentation model."""
    model = PointNet2PartSeg(num_part=num_part, normal_channel=normal_channel).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_partseg(
    obj_dir: str,
    category: str,
    checkpoint_dir: str = 'data/ckpts/partseg',
    data_root: str = 'data',
    batch_size: int = 32,
    num_points: int = 2048,
    num_votes: int = 3,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate part segmentation accuracy and mIoU.
    
    Args:
        obj_dir: Directory containing predicted .obj files
        category: Category name ('Chair', 'Table', 'Storagefurniture')
        checkpoint_dir: Directory containing model checkpoints
        data_root: Root directory for data files (should contain shapenet_partseg/, anno_id2model_id_*.json)
        batch_size: Batch size for evaluation
        num_points: Number of points to sample
        num_votes: Number of votes for prediction
        device: Device to run evaluation on
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing accuracy and mIoU metrics
    """
    # Get segmentation classes for this category
    seg_classes = get_seg_classes(category)
    seg_label_to_cat = build_seg_label_to_cat(seg_classes)
    
    # Determine model parameters based on category
    if category.lower() == 'storagefurniture':
        num_classes = 1
        num_part = 7
        checkpoint_name = 'storagefurniture'
        # For storagefurniture, we only evaluate 3 main part classes
        eval_part_indices = [0, 1, 2]
    else:
        num_classes = 16
        num_part = 50
        checkpoint_name = category.lower()
        eval_part_indices = None
    
    # Load model
    ckpt_dir = Path(checkpoint_dir) / checkpoint_name
    ckpt_path = ckpt_dir / 'best_model.pth'
    
    if not ckpt_path.exists():
        if verbose:
            print(f"Checkpoint not found: {ckpt_path}")
        return {'accuracy': 0.0, 'class_avg_iou': 0.0, 'instance_avg_iou': 0.0}
    
    model = load_partseg_model(str(ckpt_path), num_part=num_part, device=device)
    
    # Create dataset and dataloader
    dataset = PartSegDataset(
        obj_dir=obj_dir,
        category=category,
        n_points=num_points,
        data_root=data_root,
    )
    
    if len(dataset) == 0:
        if verbose:
            print(f"No valid samples found for {category}")
        return {'accuracy': 0.0, 'class_avg_iou': 0.0, 'instance_avg_iou': 0.0}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Evaluation
    total_correct = 0
    total_seen = 0
    total_seen_class = [0] * num_part
    total_correct_class = [0] * num_part
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f'Evaluating {category}') if verbose else dataloader
        
        for points, label, target, obj_ids in iterator:
            cur_batch_size, num_point, _ = points.size()
            points = points.float().to(device)
            label = label.long().to(device)
            target = target.long().to(device)
            
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(cur_batch_size, num_point, num_part).to(device)
            
            for _ in range(num_votes):
                seg_pred, _ = model(points, to_categorical(label, num_classes))
                vote_pool += seg_pred
            
            seg_pred = vote_pool / num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, num_point)).astype(np.int32)
            target_np = target.cpu().data.numpy()
            
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target_np[i, 0]]
                logits = cur_pred_val_logits[i]
                cur_pred_val[i] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            
            correct = np.sum(cur_pred_val == target_np)
            total_correct += correct
            total_seen += cur_batch_size * num_point
            
            for l in range(num_part):
                total_seen_class[l] += np.sum(target_np == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target_np == l))
            
            for i in range(cur_batch_size):
                segp = cur_pred_val[i]
                segl = target_np[i]
                cat = seg_label_to_cat[segl[0]]
                part_ious = []
                for l in seg_classes[cat]:
                    if np.sum(segl == l) == 0 and np.sum(segp == l) == 0:
                        part_ious.append(1.0)
                    else:
                        iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                        part_ious.append(iou)
                
                # Apply part index filtering for storagefurniture
                if eval_part_indices is not None:
                    part_ious = np.array(part_ious)[np.array(eval_part_indices)]
                
                shape_ious[cat].append(np.mean(part_ious))
    
    # Calculate final metrics
    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        if shape_ious[cat]:
            shape_ious[cat] = np.mean(shape_ious[cat])
        else:
            shape_ious[cat] = 0.0
    
    metrics = {
        'accuracy': total_correct / float(total_seen) if total_seen > 0 else 0.0,
        'class_avg_accuracy': np.mean([
            total_correct_class[l] / total_seen_class[l] 
            for l in range(num_part) if total_seen_class[l] > 0
        ]) if any(total_seen_class) else 0.0,
        'class_avg_iou': np.mean(list(shape_ious.values())) if shape_ious else 0.0,
        'instance_avg_iou': np.mean(all_shape_ious) if all_shape_ious else 0.0,
    }
    
    if verbose:
        print(f"\n{category} Part Segmentation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Class Avg Accuracy: {metrics['class_avg_accuracy']:.4f}")
        print(f"  Class Avg mIoU: {metrics['class_avg_iou']:.4f}")
        print(f"  Instance Avg mIoU: {metrics['instance_avg_iou']:.4f}")
    
    return metrics


def evaluate_all_categories(
    results_dir: str,
    categories: List[str] = ['Chair', 'Table', 'Storagefurniture'],
    checkpoint_dir: str = 'data/ckpts/partseg',
    data_root: str = 'data',
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate part segmentation for all categories.
    
    Args:
        results_dir: Base directory containing category subdirectories with .obj files
        categories: List of categories to evaluate
        checkpoint_dir: Directory containing model checkpoints
        data_root: Root directory for data files
        **kwargs: Additional arguments passed to evaluate_partseg
        
    Returns:
        Dictionary mapping category names to their metrics
    """
    all_metrics = {}
    
    for category in categories:
        obj_dir = Path(results_dir) / category.lower()
        if obj_dir.exists():
            metrics = evaluate_partseg(
                obj_dir=str(obj_dir),
                category=category,
                checkpoint_dir=checkpoint_dir,
                data_root=data_root,
                **kwargs
            )
            all_metrics[category] = metrics
        else:
            print(f"Results directory not found for {category}: {obj_dir}")
            all_metrics[category] = {'accuracy': 0.0, 'class_avg_iou': 0.0, 'instance_avg_iou': 0.0}
    
    return all_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Part Segmentation Evaluation')
    parser.add_argument('--obj_dir', type=str, required=True, help='Directory containing .obj files')
    parser.add_argument('--category', type=str, default='Chair', 
                       choices=['Chair', 'Table', 'Storagefurniture'],
                       help='Category to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default='data/ckpts/partseg',
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for data files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points')
    parser.add_argument('--num_votes', type=int, default=3, help='Number of votes')
    
    args = parser.parse_args()
    
    metrics = evaluate_partseg(
        obj_dir=args.obj_dir,
        category=args.category,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_votes=args.num_votes,
    )
