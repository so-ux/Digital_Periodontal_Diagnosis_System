from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch
import torch.nn as nn
import src.pointnet2.seq as pt_seq
import src.metrics.metrics as metrics

from src.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
from tqdm import tqdm

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    numerator = 2. * torch.sum(y_pred * y_true, dim=1)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), dim=1)

    return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch


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

        self.global_feat = PointnetSAModule(mlp=[c_out_2, 512, 512], use_xyz=use_xyz)

        self.FC_layer_cls = (
            pt_seq.Seq.builder(512)
                  .fc(256, bn=True)
                  .dropout(0.5)
                  .fc(33, activation=None)
        )

        self.FC_layer_seg2cls = (
            pt_seq.Seq.builder(512 + 128 + 1)
                  .conv1d(256, bn=True)
                  .dropout(0.5)
                  .conv1d(128, bn=True)
                  .conv1d(9, activation=None)
        )

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

        seed_xyz = l_xyz[-2]
        seed_features = l_features[-2]
        global_seed, global_feat = self.global_feat(seed_xyz, seed_features)
        # pred_kp = self.FC_layer_cls(global_feat.squeeze(-1))

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # global_feat = global_feat.repeat(1, 1, l_features[0].shape[2])
        # cls_features = torch.cat((l_features[0], global_feat), dim=1)

        # seg decides a confidence about how much a point is on a tooth
        seg = torch.sigmoid(self.FC_layer_conf(l_features[0]))

        # seg_conf_background = 1 - seg
        # seg_conf_foreground = seg.repeat(1, 8, 1)
        # seg_conf = torch.cat((seg_conf_background, seg_conf_foreground), dim=1)

        # cls defines a label for a point
        # cls = torch.softmax(self.FC_layer_seg2cls(torch.cat((cls_features, seg), dim=1)), dim=1)
        cls = self.FC_layer_cls(global_feat.squeeze(-1))

        return cls, seg.transpose(1, 2).contiguous()[:, :, 0]


def model_func(patches, labels):
    patches = patches.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    # seg = seg.to("cuda", dtype=torch.float32, non_blocking=True)
    # cls = cls.to("cuda", dtype=torch.float32, non_blocking=True)
    # centroids_unique = torch.unique(centroids, dim=0)
    seg_mask = torch.zeros(labels.size()).cuda()
    seg_mask[labels > 0] = 1

    cls, seg = model(patches)

    loss_seg = criterion(seg, seg_mask)

    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 8 + labels[
        labels > 0] % 10
    # labels[labels > 0] = labels[labels > 0] % 10
    label = torch.amax(labels, dim=1)

    # loss_seg = criterion(seg, seg_mask)
    loss_cls = nn.CrossEntropyLoss()(cls, label)

    seg_int_tensor = torch.zeros(seg.size())
    seg_int_tensor[seg > 0.5] = 1
    seg_int_tensor = seg_int_tensor.to(dtype=torch.int32).cuda()
    acc = metrics.mean_intersection_over_union(seg_int_tensor, seg_mask)

    return (cls, seg), (loss_seg, loss_cls), acc


if __name__ == '__main__':
    from src.data.TeethPatchTrainingDataset import TeethPatchDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    import os

    experiment_dir = '/IGIP/zsj/projects/miccai-3d-teeth-seg/output/experiment_refine_fc_0510/'
    writer = TensorboardUtils(experiment_dir).writer

    data_dir = '/IGIP/zsj/data/miccai/h5_patches/'
    train_set = TeethPatchDataset(data_dir, train=True)
    test_set = TeethPatchDataset(data_dir, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        #shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    print('Dataset ok')
    model = Pointnet2MSG(7)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('/IGIP/zsj/projects/miccai-3d-teeth-seg/output/experiment_refine_fc_0510/snapshots/model_130'))
    model.cuda()

    lr = 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=0, amsgrad=True
    )

    criterion = nn.BCELoss()

    colors = AnatomyColors()

    best_acc = 0

    for i in range(131, 1000):
        total_loss = 0
        total_loss_seg = 0
        total_loss_cls = 0
        total_acc = 0
        count = 0

        model.train()
        for batch_i, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # Let "P" be the amount of patches
            #   patches: [P, 4096, 6], P patches in total, each containing 4096 points
            #   labels: [P, 4096], ground truth label for each point
            #   indices: [P, 4096], selected from original teeth model point cloud that formed the patches
            #   seg: [P, 4096], predicted segmentation confidence
            #   cls: [P, 4096], predicted classification labels
            #   filename: str, filename of this model
            patches, labels, _ = batch_data

            (cls, seg), (loss_c, loss_cls), acc = model_func(patches, labels)
            loss = loss_c + loss_cls

            loss.backward()
            optimizer.step()

            # loss_c.detach()
            # loss_cls.detach()

            total_loss += loss.detach().item()
            total_loss_seg += loss_c.detach().item()
            total_loss_cls += loss_cls.detach().item()
            total_acc += acc
            count += 1

        print('Epoch {} - loss: {}'.format(i, total_loss / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        writer.add_scalar('train/loss_cls', total_loss_cls / count, i)
        writer.add_scalar('train/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_seg = 0
            total_loss_cls = 0
            total_acc = 0
            count = 0
            for batch_i, batch_data in enumerate(tqdm(test_loader)):
                patches, labels, centroids = batch_data

                (cls, seg), (loss_c, loss_cls), acc = model_func(patches, labels)

                if batch_i % 3 == 0 and batch_i > 462:
                    for ii in range(0, len(patches), 4):
                        nd_data = patches[ii].data.cpu().numpy()
                        nd_cls = cls[ii].data.cpu().numpy()
                        nd_seg = seg[ii].data.cpu().numpy()
                        nd_centroid = np.concatenate((centroids[ii, :, 0:3], np.array([[0, 0, 0, 1, 0, 0, 1]])), axis=1)
                        # nd_cls = nd_cls.transpose(1, 0)
                        nd_cls = np.argmax(nd_cls, axis=0)
                        color = np.zeros((nd_data.shape[0], 4))
                        color_seg = np.zeros((nd_data.shape[0], 4))
                        # color_gt = np.zeros((nd_data.shape[0], 4))
                        for pt in range(nd_data.shape[0]):
                            color[pt, 0:3] = colors.get_color(nd_cls, True) if nd_seg[pt] > 0.5 else np.ones((3, )) * 0.5
                            color[pt, 3] = 1
                            color_seg[pt, 0:3] = np.repeat(np.array([nd_seg[pt]]), 3)
                            color_seg[pt, 3] = 1
                            # color_gt[pt, 0:3] = colors.get_color(labels[ii][pt], True)
                            # color_gt[pt, 3] = 1

                        np.savetxt(
                            f'/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/cls_{batch_i}_{ii}.txt',
                            np.concatenate((np.concatenate((nd_data[:, 0:6], color), axis=1), nd_centroid), axis=0))
                    # np.savetxt(
                    #     f'/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/seg_{batch_i}_{ii}.txt',
                    #     np.concatenate((nd_data[:, 0:6], color_seg), axis=1))
                    # np.savetxt(
                    #     f'/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/gt_{batch_i}_{ii}.txt',
                    #     np.concatenate((nd_data[:, 0:6], color_gt), axis=1))
                    # np.savetxt(
                    #     f'/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/hm_{batch_i}_{ii}.txt',
                    #     vis_teeth_heatmap(nd_data[:, 0:6], nd_data[:, 6]))

                loss = loss_c + loss_cls

                total_loss += loss.item()
                total_loss_seg += loss_c.item()
                total_loss_cls += loss_cls.item()
                total_acc += acc
                count += 1

            print('  |-- test loss: {}'.format(total_loss / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_seg', total_loss_seg / count, i)
            writer.add_scalar('test/loss_cls', total_loss_cls / count, i)
            writer.add_scalar('test/acc', total_acc / count, i)

        if i == 60:
            lr = lr * 0.1
            optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
            # optimizer = optim.Adam(
            #     model.parameters(), lr=lr, weight_decay=0
            # )

        if i % 10 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(experiment_dir, 'snapshots', 'model_{}'.format(i)))

        if total_acc > best_acc:
            best_acc = total_acc
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(experiment_dir, 'snapshots', 'model_best'))

        writer.flush()

