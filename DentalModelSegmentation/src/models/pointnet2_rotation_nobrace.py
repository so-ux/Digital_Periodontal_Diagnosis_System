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
from src.data.TeethRotationNoBraceDataset import TeethRotationDataset

from src.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

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
            pt_seq.Seq.builder(512)
            .fc(64, bn=True)
            .dropout(0.2)
            .fc(3, activation=None)
        )

        self.FC_layer_axis_forward = (
            pt_seq.Seq.builder(512)
            .fc(64, bn=True)
            .dropout(0.2)
            .fc(3, activation=None)
        )

        self.FC_layer_conf = (
            pt_seq.Seq.builder(512 + 128)
                  .conv1d(64, bn=True)
                  .dropout(0.2)
                  .conv1d(9, activation=None)
        )

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

        axis_up = self.FC_layer_axis_up(global_feat[:, :, 0])
        axis_forward = self.FC_layer_axis_forward(global_feat[:, :, 0])
        cls_features = torch.cat((l_features[0], global_feat.repeat(1, 1, l_features[0].shape[2])), dim=1)
        seg = torch.softmax(self.FC_layer_conf(cls_features), dim=1)

        return torch.nn.functional.normalize(axis_up, p=2, dim=-1),\
               torch.nn.functional.normalize(axis_forward, p=2, dim=-1),\
               seg


class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.rot_net1 = Pointnet2MSG(3)
        self.rot_net2 = Pointnet2MSG(3)

    def forward(self, pointcloud):
        ax_up, ax_forward, _ = self.rot_net1(pointcloud)

        for ii in range(ax_up.shape[0]):
            nd_data = pointcloud[ii]
            axis_up = ax_up[ii]
            axis_forward = ax_forward[ii]

            rot_axis = torch.cross(axis_up, torch.Tensor([0, 0, 1]).cuda())
            cos_theta = torch.sum(axis_up * torch.Tensor([0, 0, 1]).cuda()) / torch.norm(axis_up)
            rot_angle = torch.arccos(torch.clip(cos_theta, -1, 1))
            rot_matrix = torch.matrix_exp(torch.cross(torch.eye(3).cuda(), -rot_axis.unsqueeze(-1).repeat(1, 3) / torch.norm(rot_axis) * rot_angle))
            nd_data[:, 0:3] = torch.matmul(rot_matrix, nd_data[:, 0:3].unsqueeze(-1))[:, :, 0]
            nd_data[:, 3:6] = torch.matmul(rot_matrix, nd_data[:, 3:6].unsqueeze(-1))[:, :, 0]
            axis_forward = torch.matmul(rot_matrix, axis_forward)

            rot_axis = torch.Tensor([0, 0, 1]).cuda()
            cos_theta = torch.sum(axis_forward * torch.Tensor([0, -1, 0]).cuda()) / torch.norm(axis_forward)
            rot_angle = torch.arccos(torch.clip(cos_theta, -1, 1))
            rot_matrix = torch.matrix_exp(torch.cross(torch.eye(3).cuda(), -rot_axis.unsqueeze(-1).repeat(1, 3) / torch.norm(rot_axis) * rot_angle))
            nd_data[:, 0:3] = torch.matmul(rot_matrix, nd_data[:, 0:3].unsqueeze(-1))[:, :, 0]
            nd_data[:, 3:6] = torch.matmul(rot_matrix, nd_data[:, 3:6].unsqueeze(-1))[:, :, 0]

            pointcloud[ii] = nd_data
        ax_up, ax_forward, pred_seg = self.rot_net2(pointcloud)
        return ax_up, ax_forward, pred_seg


def model_func(data, axes, labels):
    data = data[:, :, 0:6].to("cuda", dtype=torch.float32, non_blocking=True)
    axes = axes.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    angles = torch.zeros((len(data), ), dtype=torch.float32).cuda()

    labels[labels > 30] -= 20
    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 0 + labels[
        labels > 0] % 10

    pred_axis_up, pred_axis_forward, pred_seg = model(data)

    loss_axis_up = criterion_axis(pred_axis_up, axes[:, 0, :])
    loss_axis_forward = criterion_axis(pred_axis_forward, axes[:, 1, :])
    loss_axes_angle = criterion_angle(torch.sum(pred_axis_up * pred_axis_forward, dim=-1), angles)
    loss_seg = criterion_seg(pred_seg, labels)
    return pred_axis_up, pred_axis_forward, pred_seg, loss_axis_up, loss_axis_forward, torch.mean(loss_axes_angle), loss_seg


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    from src.vis.vis_teeth_cls import vis_teeth_cls

    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_pn2_rot_nobrace_0719/'
    writer = TensorboardUtils(experiment_dir).writer

    train_set = TeethRotationDataset(train=True)
    test_set = TeethRotationDataset(train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    model = Pointnet2MSG(3)
    model = torch.nn.DataParallel(model)
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

    for i in range(0, 1000):
        total_loss = 0
        total_loss_axis_up = 0
        total_loss_axis_forward = 0
        total_loss_axes_angle = 0
        total_loss_seg = 0
        count = 0

        model.train()
        for batch_i, batch_data in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            data, axes, labels = batch_data

            pred_axis_up, pred_axis_forward, pred_seg, loss_axis_up, loss_axis_forward, loss_axes_angle, loss_seg = model_func(data, axes, labels)

            loss = loss_axis_up + loss_axis_forward + loss_axes_angle + loss_seg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_axis_up += loss_axis_up.item()
            total_loss_axis_forward += loss_axis_forward.item()
            total_loss_axes_angle += loss_axes_angle.item()
            total_loss_seg += loss_seg.item()
            count += 1

            if i % 5 == 0 and batch_i < 1:
                color_origin = np.repeat(np.array([[0, 0.5, 0.5, 1]]), data.shape[1], axis=0)
                color_rot = np.repeat(np.array([[1, 0, 0, 1]]), data.shape[1], axis=0)
                for ii in range(16):
                    nd_data = data[ii]
                    nd_seg = labels[ii]
                    axis_up = pred_axis_up[ii].data.cpu().numpy()
                    axis_forward = pred_axis_forward[ii].data.cpu().numpy()

                    origin_data = np.concatenate((nd_data, color_origin), axis=-1)

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

                    rot_data = np.concatenate((nd_data, color_rot), axis=-1)

                    np.savetxt(
                        f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis_nobrace/train_{batch_i}_{ii}.txt', np.concatenate((origin_data, rot_data), axis=0))


        print('Epoch {} - loss: {}'.format(i, total_loss / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_axis_up', total_loss_axis_up / count, i)
        writer.add_scalar('train/loss_axis_forward', total_loss_axis_forward / count, i)
        writer.add_scalar('train/loss_axes_angle', total_loss_axes_angle / count, i)
        writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        writer.flush()

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_axis_up = 0
            total_loss_axis_forward = 0
            total_loss_axes_angle = 0
            total_loss_seg = 0
            count = 0
            for batch_i, batch_data in enumerate(tqdm.tqdm(test_loader)):
                data, axes, labels = batch_data

                pred_axis_up, pred_axis_forward, pred_seg, loss_axis_up, loss_axis_forward, loss_axes_angle, loss_seg = model_func(data, axes, labels)

                loss = loss_axis_up + loss_axis_forward + loss_axes_angle + loss_seg

                if i % 5 == 0 and batch_i < 10:
                    color_origin = np.repeat(np.array([[0, 0.5, 0.5, 1]]), data.shape[1], axis=0)
                    color_rot = np.repeat(np.array([[1, 0, 0, 1]]), data.shape[1], axis=0)
                    for ii in range(len(data)):
                        nd_data = data[ii]
                        nd_seg = torch.argmax(pred_seg, dim=1)[ii].data.cpu().numpy()
                        axis_up = pred_axis_up[ii].data.cpu().numpy()
                        axis_forward = pred_axis_forward[ii].data.cpu().numpy()

                        origin_data = np.concatenate((nd_data, color_origin), axis=-1)

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

                        #for pt in range(nd_data.shape[0]):
                            #color_rot[pt, 0:3] = vis_teeth_cls(nd_data, nd_seg)
                        #rot_data = np.concatenate((nd_data, color_rot), axis=-1)
                        rot_data = vis_teeth_cls(nd_data, nd_seg)

                        np.savetxt(
                            f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis_nobrace/{batch_i}_{ii}.txt', np.concatenate((origin_data, rot_data), axis=0))
                        #np.savetxt(
                            #f'/IGIP/zsj/projects/miccai-3d-teeth-seg/data/data_rot/vis/{batch_i}_{ii}.txt', origin_data)

                total_loss += loss.item()
                total_loss_axis_up += loss_axis_up.item()
                total_loss_axis_forward += loss_axis_forward.item()
                total_loss_axes_angle += loss_axes_angle.item()
                total_loss_seg += loss_seg.item()
                count += 1
            print('  Testing loss: {}'.format(total_loss / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_axis_up', total_loss_axis_up / count, i)
            writer.add_scalar('test/loss_axis_forward', total_loss_axis_forward / count, i)
            writer.add_scalar('test/loss_axes_angle', total_loss_axes_angle / count, i)
            writer.add_scalar('test/loss_seg', total_loss_seg / count, i)
            writer.flush()

        if i % 100 == 0:
            lr = lr * 0.5
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=0, amsgrad=True
            )

        if i % 20 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            #torch.save(model.module.state_dict(),
                       #os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_{}'.format(i))))

        if best_loss > total_loss / count:
            best_loss = total_loss / count
            #torch.save(model.module.state_dict(),
                       #os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_best')))
