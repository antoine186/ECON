# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import pytorch_lightning as pl
import torch
from termcolor import colored

from ..dataset.mesh_util import *
from ..net.geometry import orthogonal


class Format:
    end = '\033[0m'
    start = '\033[4m'

# Initializes a dictionary of different loss functions used during the optimization of a model. Each loss type has a specific weight and value that contribute to the overall objective function.
def init_loss():

    losses = {
    # Cloth: chamfer distance
    # 1e3 means this loss has a significant influence (high weight) on the overall optimization.
    # Value: Initially set to 0.0, this will accumulate the actual computed loss during training.
        "cloth": {"weight": 1e3, "value": 0.0},
    # Stiffness: This loss penalizes the difference between rotation transformations of connected vertices, keeping the cloth stiffness intact. It's often used to maintain the rigidity of edges in a mesh.
    # 1e5 gives this loss a very high influence, emphasizing cloth stiffness during the optimization.
        "stiff": {"weight": 1e5, "value": 0.0},
    # Cloth: det(R) = 1
    # Rigid: Ensures that the rotation matrix R applied to the cloth has a determinant of 1, which means it preserves volume and avoids deformation.
    # 1e5, giving this constraint a high priority in the optimization process.
        "rigid": {"weight": 1e5, "value": 0.0},
    # Cloth: edge length
    # Loss is disabled.
        "edge": {"weight": 0, "value": 0.0},
    # Cloth: normal consistency
    # Loss is disabled.
        "nc": {"weight": 0, "value": 0.0},
    # Cloth: laplacian smoonth
    # Laplacian Smoothness (lapla): This loss smooths the cloth mesh by penalizing differences between a vertex and the average of its neighboring vertices.
    # 1e2, giving it moderate influence, which helps smooth the mesh while retaining other properties.
        "lapla": {"weight": 1e2, "value": 0.0},
    # Body: Normal_pred - Normal_smpl
        "normal": {"weight": 1e0, "value": 0.0},
    # Body: Silhouette_pred - Silhouette_smpl
    # Naming not consistent with paper. It corresponds to depth loss Ld.
        "silhouette": {"weight": 1e0, "value": 0.0},
    # Joint: reprojected joints difference
    # This is LJ_diff.
        "joint": {"weight": 1e0, "value": 0.0},
    }

    return losses


class SubTrainer(pl.Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        """Save model/training states as a checkpoint file through state-dump and file-write.
        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        _checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)

        del_keys = []
        for key in _checkpoint["state_dict"].keys():
            for ignore_key in ["normal_filter", "voxelization", "reconEngine"]:
                if ignore_key in key:
                    del_keys.append(key)
        for key in del_keys:
            del _checkpoint["state_dict"][key]

        pl.utilities.cloud_io.atomic_save(_checkpoint, filepath)


def query_func(opt, netG, features, points, proj_matrix=None):
    """
        - points: size of (bz, N, 3)
        - proj_matrix: size of (bz, 4, 4)
    return: size of (bz, 1, N)
    """
    assert len(points) == 1
    samples = points.repeat(opt.num_views, 1, 1)
    samples = samples.permute(0, 2, 1)    # [bz, 3, N]

    # view specific query
    if proj_matrix is not None:
        samples = orthogonal(samples, proj_matrix)

    calib_tensor = torch.stack([torch.eye(4).float()], dim=0).type_as(samples)

    preds = netG.query(
        features=features,
        points=samples,
        calibs=calib_tensor,
        regressor=netG.if_regressor,
    )

    if type(preds) is list:
        preds = preds[0]

    return preds


def query_func_IF(batch, netG, points):
    """
        - points: size of (bz, N, 3)
    return: size of (bz, 1, N)
    """

    batch["samples_geo"] = points
    batch["calib"] = torch.stack([torch.eye(4).float()], dim=0).type_as(points)

    preds = netG(batch)

    return preds.unsqueeze(1)


def batch_mean(res, key):
    return torch.stack([
        x[key] if torch.is_tensor(x[key]) else torch.as_tensor(x[key]) for x in res
    ]).mean()


def accumulate(outputs, rot_num, split):

    hparam_log_dict = {}

    metrics = outputs[0].keys()
    datasets = split.keys()

    for dataset in datasets:
        for metric in metrics:
            keyword = f"{dataset}/{metric}"
            if keyword not in hparam_log_dict.keys():
                hparam_log_dict[keyword] = 0
            for idx in range(split[dataset][0] * rot_num, split[dataset][1] * rot_num):
                hparam_log_dict[keyword] += outputs[idx][metric].item()
            hparam_log_dict[keyword] /= (split[dataset][1] - split[dataset][0]) * rot_num

    print(colored(hparam_log_dict, "green"))

    return hparam_log_dict
