"""
使用Raw PointNet++分割牙齿并分类

该网络作为粗分割网络，训练到大致能够给予每颗牙齿标签即可。
"""
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch
import torch.nn as nn
import tqdm
from scipy import linalg

import src.pointnet2.seq as pt_seq
from src.data.TeethRotationDataset import TeethRotationDataset

from src.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
from src.utils.cfdp import get_rotation_axis

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=6, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1, 0.2],
                nsamples=[16 * 2, 32 * 2, 64 * 2],
                mlps=[[c_in, 16, 16, 32, 64], [c_in, 32, 32, 64, 64], [c_in, 32, 32, 64, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 64 + 64 + 128

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2, 0.3],
                nsamples=[16, 32, 64],
                mlps=[[c_in, 64, 64, 64, 128], [c_in, 64, 96, 96, 128], [c_in, 64, 96, 96, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128 + 256

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4, 0.6],
                nsamples=[16, 32, 64],
                mlps=[[c_in, 128, 196, 196, 256], [c_in, 128, 196, 196, 256], [c_in, 128, 196, 196, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256 + 512

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8, 1.0],
                nsamples=[16, 32, 64],
                mlps=[[c_in, 256, 256, 256, 512], [c_in, 256, 384, 384, 512], [c_in, 256, 384, 384, 1024]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512 + 1024

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.global_feat = PointnetSAModule(mlp=[c_out_3, 512, 512], use_xyz=use_xyz)

        self.FC_layer_axis_up = (
            pt_seq.Seq.builder(512 + 128)
            .conv1d(64, bn=True)
            .dropout(0.1)
            .conv1d(3, activation=None)
        )

        self.FC_layer_axis_forward = (
            pt_seq.Seq.builder(512 + 128)
            .conv1d(64, bn=True)
            .dropout(0.1)
            .conv1d(3, activation=None)
        )

        #self.FC_layer_conf = (
            #pt_seq.Seq.builder(512 + 128)
                  #.conv1d(64, bn=True)
                  #.dropout()
                  #.conv1d(9, activation=None)
        #)

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        seed_xyz = l_xyz[-1]
        seed_features = l_features[-1]
        global_seed, global_feat = self.global_feat(seed_xyz, seed_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

#        axis_up = self.FC_layer_axis_up(global_feat[:, :, 0])
#        axis_forward = self.FC_layer_axis_forward(global_feat[:, :, 0])
        cls_features = torch.cat((l_features[0], global_feat.repeat(1, 1, l_features[0].shape[2])), dim=1)
#        seg = torch.softmax(self.FC_layer_conf(cls_features), dim=1)

        axis_up = self.FC_layer_axis_up(cls_features)
        axis_forward = self.FC_layer_axis_forward(cls_features)

#        axis_up = axis_up.transpose(2, 1)
        #axis_forward = axis_forward.transpose(2, 1)

        # return torch.nn.functional.normalize(torch.mean(axis_up, dim=-1), p=2, dim=-1),\
        #        torch.nn.functional.normalize(torch.mean(axis_forward, dim=-1), p=2, dim=-1)
        return torch.nn.functional.normalize(axis_up, p=2, dim=1),\
               torch.nn.functional.normalize(axis_forward, p=2, dim=1)


class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.rot_net = Pointnet2MSG(3)
        self.rot_net.load_state_dict(torch.load('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/model/model_rotation'))

    def forward(self, data):
        pred_axis_up, pred_axis_forward = self.rot_net(data)
        return pred_axis_up, pred_axis_forward

        for ii in range(len(data)):
            nd_data = data[ii].data.cpu().numpy()
            axis_up = pred_axis_up[ii].data.cpu().numpy()
            axis_forward = pred_axis_forward[ii].data.cpu().numpy()

            rot_axis = np.cross(axis_up, np.array([0, 0, 1]))
            cos_theta = np.sum(axis_up * np.array([0, 0, 1])) / np.linalg.norm(axis_up)
            rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
            rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
            nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
            nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
            axis_forward = np.matmul(rot_matrix, axis_forward)

            rot_axis = np.array([0, 0, 1])
            cos_theta = np.sum(axis_forward * np.array([0, -1, 0])) / np.linalg.norm(axis_forward)
            rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
            rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
            nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
            nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]

            data[ii] = torch.from_numpy(nd_data).cuda()

        return self.rot_net(data)


def calc_angle(ax0, ax1):
    result = []
    for b in range(len(ax0)):
        result.append(180 * torch.arccos(torch.clip(torch.dot(ax0[b], ax1[b]) / (torch.sum(ax0[b] ** 2) * torch.sum(ax1[b] ** 2)), -1, 1)).item() / torch.pi)
#    result /= len(ax0)
    return result


def model_func(data, axes, labels):
    data = data[:, :, 0:6].to("cuda", dtype=torch.float32, non_blocking=True)
    axes = axes.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    angles = torch.zeros((len(data), ), dtype=torch.float32).cuda()

    labels[labels > 30] -= 20
    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 0 + labels[
        labels > 0] % 10

    pred_axis_up, pred_axis_forward = model(data)
    pred_axis_up = pred_axis_up.transpose(2, 1).contiguous()
    pred_axis_forward = pred_axis_forward.transpose(2, 1).contiguous()

    clus_axis_up = []
    clus_axis_forward = []
    for ii in range(len(pred_axis_up)):
        clus_axis_up.append(get_rotation_axis(np.asarray(pred_axis_up[ii].data.cpu().numpy(), dtype=np.float64)))
        clus_axis_forward.append(get_rotation_axis(np.asarray(pred_axis_forward[ii].data.cpu().numpy(), dtype=np.float64)))

    pred_axis_up = torch.from_numpy(np.array(clus_axis_up)).to('cuda', dtype=torch.float32)
    pred_axis_forward = torch.from_numpy(np.array(clus_axis_forward)).to('cuda', dtype=torch.float32)

    pred_seg = 0
    # loss_axis_up = 0
    # loss_axis_forward = 0
    # loss_axes_angle = torch.Tensor(0)

    loss_axis_up = criterion_axis(pred_axis_up, axes[:, 0, :])
    loss_axis_forward = criterion_axis(pred_axis_forward, axes[:, 1, :])
    loss_axes_angle = criterion_angle(torch.sum(pred_axis_up * pred_axis_forward, dim=-1), angles)

    error_up = calc_angle(pred_axis_up, axes[:, 0, :])
    error_forward = calc_angle(pred_axis_forward, axes[:, 1, :])
    # error_up = 0
    # error_forward = 0
    return pred_axis_up, pred_axis_forward, pred_seg, loss_axis_up, loss_axis_forward, torch.mean(loss_axes_angle), error_up, error_forward


def draw_cloud_and_axes(filename, pc0, pred_axis_up, pred_axis_forward):
    with open(filename, 'w') as fp:
        # fp.write(f'v 0 0 0 0 0 0\nvn 0 0 1\n')
        for i in range(len(pc0)):
            fp.write(f'v {pc0[i, 0]} {pc0[i, 1]} {pc0[i, 2]} 0.5 0.5 0.5\n')
            # fp.write(f'vn {pc0[i, 3]} {pc0[i, 4]} {pc0[i, 5]}\n')
            # fp.write(f'v {pc0[i, 0] + 5 * pred_axis_up[i, 0]} {pc0[i, 1] + 5 * pred_axis_up[i, 1]} {pc0[i, 2] + 5 * pred_axis_up[i, 2]} 1 1 1\n')
            fp.write(f'v {pc0[i, 0] + 5 * pred_axis_forward[i, 0]} {pc0[i, 1] + 5 * pred_axis_forward[i, 1]} {pc0[i, 2] + 5 * pred_axis_forward[i, 2]} 1 0 0\n')
            fp.write(f'v {pc0[i, 0] + 5 * pred_axis_forward[i, 0]} {pc0[i, 1] + 5 * pred_axis_forward[i, 1]} {pc0[i, 2] + 5 * pred_axis_forward[i, 2]} 1 0 0\n')
            fp.write(f'l {i * 3 + 1} {i * 3 + 2}\n')
            fp.write(f'l {i * 3 + 1} {i * 3 + 3}\n')
        for i in range(len(pc0)):
            fp.write(f'v {40 * pred_axis_forward[i, 0]} {40 * pred_axis_forward[i, 1]} {40 * pred_axis_forward[i, 2]} 0 1 0\n')
        fp.close()


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    from src.vis.vis_teeth_cls import vis_teeth_cls

    # train_set = TeethRotationDataset(train=True)
    test_set = TeethRotationDataset(train=False)
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=32,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=8
    # )
    test_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    model = RotationNet()
    model.cuda()

    lr = 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=0, amsgrad=True
    )

    criterion_axis = nn.MSELoss()
    criterion_angle = nn.SmoothL1Loss()
    criterion_seg = nn.CrossEntropyLoss()

    colors = AnatomyColors()

    best_loss = 1e9

    with torch.no_grad():
        model.eval()
        total_loss = 0
        total_loss_axis_up = 0
        total_loss_axis_forward = 0
        total_loss_axes_angle = 0
        total_loss_seg = 0
        total_error_up = 0
        total_error_forward = 0
        max_error_up = 0
        max_error_forward = 0
        count = 0
        for batch_i, batch_data in enumerate(tqdm.tqdm(test_loader)):
            data, axes, labels = batch_data

            pred_axis_up, pred_axis_forward, pred_seg, loss_axis_up, loss_axis_forward, loss_axes_angle, error_up, error_forward = model_func(
                data, axes, labels)
            # pred_axis_up, pred_axis_forward, pred_seg, loss_axis_up, loss_axis_forward, loss_axes_angle = model_func(data, axes, labels)

            loss = loss_axis_up + loss_axis_forward + loss_axes_angle

            print(error_up, error_forward)

            if batch_i < 10:
                color_origin = np.repeat(np.array([[0, 0.5, 0.5, 1]]), 9600, axis=0)
                color_rot = np.repeat(np.array([[1, 0, 0, 1]]), 9600, axis=0)
                for ii in range(len(data)):
                    nd_data = data[ii]
                    axis_up = pred_axis_up[ii].data.cpu().numpy()
                    axis_forward = pred_axis_forward[ii].data.cpu().numpy()

                    # origin_data = np.concatenate((nd_data, color_origin), axis=-1)
                    # origin_data = np.array(nd_data)
                    # origin_axes = np.array(axes[ii].data.to("cpu").numpy())

                    rot_axis = np.cross(axis_up, np.array([0, 0, 1]))
                    cos_theta = np.sum(axis_up * np.array([0, 0, 1])) / np.linalg.norm(axis_up)
                    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
                    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
                    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
                    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
                    axis_forward = np.matmul(rot_matrix, axis_forward)
                    axis_up = np.matmul(rot_matrix, axis_up)
                    # origin_axes = np.matmul(rot_matrix, origin_axes[:, :, np.newaxis])[:, :, 0]

                    rot_axis = np.array([0, 0, 1])
                    cos_theta = np.sum(axis_forward * np.array([0, -1, 0])) / np.linalg.norm(axis_forward)
                    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
                    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
                    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
                    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
                    axis_forward = np.matmul(rot_matrix, axis_forward)
                    axis_up = np.matmul(rot_matrix, axis_up)
                    # origin_axes = np.matmul(rot_matrix, origin_axes[:, :, np.newaxis])[:, :, 0]

                    # draw_cloud_and_axes(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis_dense/{batch_i}_{ii}.obj', nd_data, axis_up, axis_forward)
                    # np.savetxt(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis_dense/{batch_i}_{ii}.txt', axis_forward)
                    np.savetxt(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis_dense/{batch_i}_{ii}.xyz', nd_data)

            total_loss += loss.item()
            total_loss_axis_up += loss_axis_up.item()
            total_loss_axis_forward += loss_axis_forward.item()
            total_loss_axes_angle += loss_axes_angle.item()
            total_error_up += np.mean(error_up)
            total_error_forward += np.mean(error_forward)
            max_error_up = max(max_error_up, abs(np.max(error_up)))
            max_error_forward = max(max_error_forward, abs(np.max(error_forward)))
            count += 1
        print('loss: {}'.format(total_loss / count))
        print('loss_axis_up: {}'.format(total_loss_axis_up / count))
        print('loss_axis_forward: {}'.format(total_loss_axis_forward / count))
        print('loss_axes_angle: {}'.format(total_loss_axes_angle / count))
        print('angle_error_up: {}'.format(total_error_up / count))
        print('angle_error_forward: {}'.format(total_error_forward / count))
        print('max_error_up: {}'.format(max_error_up))
        print('max_error_forward: {}'.format(max_error_forward))

