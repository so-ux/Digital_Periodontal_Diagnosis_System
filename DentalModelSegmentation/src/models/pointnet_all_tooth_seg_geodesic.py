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

import src.pointnet2.seq as pt_seq

from src.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors
from src.metrics.metrics import mean_intersection_over_union


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
                npoint=2048,
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
                npoint=1024,
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
                npoint=512,
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
                npoint=128,
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

        # self.FC_layer_cls = (
        #     pt_seq.Seq.builder(512 + 128)
        #           .conv1d(256, bn=True)
        #           .dropout()
        #           .conv1d(33, activation=None)
        # )

        self.FC_layer_conf = (
            pt_seq.Seq.builder(128)
                  .conv1d(64, bn=True)
                  .dropout()
                  .conv1d(1, activation=None)
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

        # global_feat = global_feat.repeat(1, 1, l_features[0].shape[2])
        # cls_features = torch.cat((l_features[0], global_feat), dim=1)

        # seg decides a confidence about how much a point is on a tooth
        seg = torch.sigmoid(self.FC_layer_conf(l_features[0]))
        # seg = torch.sigmoid(self.FC_layer_conf(l_features[0]))

        # seg_conf_background = 1 - seg
        # seg_conf_foreground = seg.repeat(1, 32, 1)
        # seg_conf = torch.cat((seg_conf_background, seg_conf_foreground), dim=1)

        # cls defines a label for a point
        # cls = torch.softmax(self.FC_layer_cls(cls_features) * seg_conf, dim=1)

        return seg.transpose(1, 2).contiguous()[:, :, 0]



def extract_centroids(pred, data):
    """
    Extract centroids from prediction tensor

    :param pred: Tensor [B, N]
    :type pred: torch.Tensor
    :return: Tensor [B, N, 3]
    :rtype: torch.Tensor
    """

    # Due to normalization, coordinate smaller than -1 is illegal,
    # so -100 is chosen to represent non-tooth vertex
    # See: TeethTrainingDataset.py
    all_centroids = -100 * torch.ones((data.size(0), data.size(1), 3))
    for bi in range(pred.size(0)):
        for ti in torch.unique(pred[bi]):
            all_centroids[bi, ti, :] = torch.mean(data[bi, pred[bi] == ti, 0:3], dim=0)
    return all_centroids


def chamfer_distance_without_batch(p1, p2, debug=False):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: Whether you need to output debug info
    :return: Sum of Chamfer Distance of two point sets
    """

    if len(p1.shape) == 2:
        p1 = p1.unsqueeze(0)

    if len(p2.shape) == 2:
        p2 = p2.unsqueeze(0)

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])
    p1 = p1.repeat(p2.size(1), 1, 1)

    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist = torch.norm(dist, 2, dim=2)

    dist_sort, indices = torch.sort(dist)
    # print(dist_sort.shape, indices.shape)
    dist1 = dist_sort[:, 0]
    index_min_dis = indices[:, 0]

    return dist1, index_min_dis


def model_func(data, labels):
    data = data.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.float32, non_blocking=True)

    seg = model(data)

    loss_seg = criterion(seg, labels)

    return seg, loss_seg, 0


if __name__ == '__main__':
    import os
    from src.data.TeethTrainingDataset import TeethTrainingDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np

    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_pn2_gd_0523/'
    writer = TensorboardUtils(experiment_dir).writer

    data_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/'
    train_set = TeethTrainingDataset(os.path.join(data_dir, 'train.list'), train=True, require_dicts=True)
    test_set = TeethTrainingDataset(os.path.join(data_dir, 'test.list'), train=False, require_dicts=True)
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=False,
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

    model = Pointnet2MSG(4)
    model.cuda()

    lr = 1e-3
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=0, amsgrad=True
    )

    criterion = nn.MSELoss()

    colors = AnatomyColors()

    best_acc = 1000

    for i in range(0, 1000):
        total_loss = 0
        total_loss_seg = 0
        total_loss_cls = 0
        total_acc = 0
        count = 0

        model.train()
        for batch_i, batch_data in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            data, labels, centroids, real_size, gd = batch_data

            gd = np.exp(-(gd ** 2) / (2 * 0.09))

            seg, loss, acc = model_func(data, gd)
            # (cls, seg), (loss_c, loss_cls, loss_cd) = model_func(data, labels, centroids)
            # loss = loss_c + loss_cls #+ loss_cd

            loss.backward()
            optimizer.step()

            # loss_c.detach()
            # loss_cls.detach()

            total_loss += loss.item()
            total_acc += acc
            # total_loss_seg += loss_c.item()
            # total_loss_cls += loss_cls.item()
            count += 1

        print('Epoch {} - loss: {}, acc: {}'.format(i, total_loss / count, total_acc / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        # writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        # writer.add_scalar('train/loss_cls', total_loss_cls / count, i)
        writer.add_scalar('train/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_seg = 0
            total_loss_cls = 0
            total_acc = 0
            count = 0
            for batch_i, batch_data in enumerate(tqdm.tqdm(test_loader)):
                data, labels, centroids, real_size, gd = batch_data

                gd = np.exp(-(gd ** 2) / (2 * 0.09))

                seg, loss, acc = model_func(data, gd)

                if i % 5 == 0:
                    for ii in range(len(data)):
                        nd_data = data[ii].data.cpu().numpy()
                        nd_seg = seg[ii].data.cpu().numpy()
                        color_seg = np.zeros((nd_data.shape[0], 4))
                        for pt in range(nd_data.shape[0]):
                            color_seg[pt, 0:3] = np.repeat(np.array([nd_seg[pt]]), 3)
                            color_seg[pt, 3] = 1

                        np.savetxt(
                            '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_pn_ct/{}_{}.seg.txt'.format(batch_i,
                                                                                                                ii),
                            np.concatenate((nd_data[:, 0:6], color_seg), axis=1))

                total_loss += loss.item()
                total_acc += acc
                count += 1
            print('  |-- test loss: {}, acc: {}'.format(total_loss / count, total_acc / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/acc', total_acc / count, i)

        if i % 100 == 0:
            lr = lr * 0.5
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=0, amsgrad=True
            )

        if i % 20 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_{}'.format(i))))

        if best_acc > total_loss / count:
            best_acc = total_loss / count
            torch.save(model.state_dict(),
                       os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_best')))
