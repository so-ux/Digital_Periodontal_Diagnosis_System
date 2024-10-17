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

from src.data.TeethRotationDataset import TeethRotationDataset
from src.models.pct_models import Pct, PctDense

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors


def calc_angle(ax0, ax1):
    result = []
    for b in range(len(ax0)):
        ang = 180 * torch.arccos(torch.clip(torch.dot(ax0[b], ax1[b]) / (torch.sum(ax0[b] ** 2) * torch.sum(ax1[b] ** 2)), -1, 1)).item() / torch.pi
        result.append(abs(ang))
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

    pred_axis_up, pred_axis_forward = model(data.permute(0, 2, 1))

    loss_axis_up = criterion_axis(pred_axis_up, axes[:, 0, :])
    loss_axis_forward = criterion_axis(pred_axis_forward, axes[:, 1, :])
    loss_axes_angle = criterion_angle(torch.sum(pred_axis_up * pred_axis_forward, dim=-1), angles)
    # loss_seg = criterion_seg(pred_seg, labels)

    error_up = calc_angle(pred_axis_up, axes[:, 0, :])
    error_forward = calc_angle(pred_axis_forward, axes[:, 1, :])

    return pred_axis_up, pred_axis_forward, loss_axis_up, loss_axis_forward, torch.mean(loss_axes_angle), error_up, error_forward


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    from src.vis.vis_teeth_cls import vis_teeth_cls
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0.2, required=False)
    args = parser.parse_args()

    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_pct_rot_dense_0720/'
    writer = TensorboardUtils(experiment_dir).writer

    train_set = TeethRotationDataset(train=True)
    test_set = TeethRotationDataset(train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=8,
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

    model = PctDense(args)
    model = torch.nn.DataParallel(model)
    model.cuda()

    lr = 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )

    criterion_axis = nn.MSELoss()
    criterion_angle = nn.SmoothL1Loss()
    # criterion_seg = nn.CrossEntropyLoss()

    colors = AnatomyColors()

    best_loss = 1e9

    for i in range(0, 1000):
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

        model.train()
        for batch_i, batch_data in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            data, axes, labels = batch_data

            pred_axis_up, pred_axis_forward, loss_axis_up, loss_axis_forward, loss_axes_angle, error_up, error_forward = model_func(data, axes, labels)

            loss = loss_axis_up + loss_axis_forward + loss_axes_angle
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_axis_up += loss_axis_up.item()
            total_loss_axis_forward += loss_axis_forward.item()
            total_loss_axes_angle += loss_axes_angle.item()
            total_error_up += np.mean(error_up)
            total_error_forward += np.mean(error_forward)
            max_error_up = max(max_error_up, abs(np.max(error_up)))
            max_error_forward = max(max_error_forward, abs(np.max(error_forward)))
            # total_loss_seg += loss_seg.item()
            count += 1

            if i % 5 == 0 and batch_i < 1:
                color_origin = np.repeat(np.array([[0, 0.5, 0.5, 1]]), data.shape[1], axis=0)
                color_rot = np.repeat(np.array([[1, 0, 0, 1]]), data.shape[1], axis=0)
                for ii in range(len(data)):
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
                        f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis/train_{batch_i}_{ii}.txt', np.concatenate((origin_data, rot_data), axis=0))


        print('Epoch {} - loss: {}'.format(i, total_loss / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_axis_up', total_loss_axis_up / count, i)
        writer.add_scalar('train/loss_axis_forward', total_loss_axis_forward / count, i)
        writer.add_scalar('train/loss_axes_angle', total_loss_axes_angle / count, i)
        writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        writer.add_scalar('train/angle_error_up', total_error_up / count, i)
        writer.add_scalar('train/angle_error_forward', total_error_forward / count, i)
        writer.add_scalar('train/max_error_up', max_error_up, i)
        writer.add_scalar('train/max_error_forward', max_error_forward, i)
        writer.flush()

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

                pred_axis_up, pred_axis_forward, loss_axis_up, loss_axis_forward, loss_axes_angle, error_up, error_forward = model_func(data, axes, labels)

                loss = loss_axis_up + loss_axis_forward + loss_axes_angle

                if i % 5 == 0 and batch_i < 10:
                    color_origin = np.repeat(np.array([[0, 0.5, 0.5, 1]]), data.shape[1], axis=0)
                    color_rot = np.repeat(np.array([[1, 0, 0, 1]]), data.shape[1], axis=0)
                    for ii in range(len(data)):
                        nd_data = data[ii]
                        # nd_seg = torch.argmax(pred_seg, dim=1)[ii].data.cpu().numpy()
                        axis_up = pred_axis_up[ii].data.cpu().numpy()
                        axis_forward = pred_axis_forward[ii].data.cpu().numpy()

                        #origin_data = np.concatenate((nd_data, color_origin), axis=-1)
                        origin_data = vis_teeth_cls(nd_data, labels[ii])

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
                        rot_data = np.concatenate((nd_data, color_rot), axis=-1)
                        # rot_data = vis_teeth_cls(nd_data, nd_seg)

                        np.savetxt(
                            f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis/{batch_i}_{ii}.txt', np.concatenate((origin_data, rot_data), axis=0))
                        # np.savetxt(
                        #     f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/vis/{batch_i}_{ii}.txt', origin_data)

                total_loss += loss.item()
                total_loss_axis_up += loss_axis_up.item()
                total_loss_axis_forward += loss_axis_forward.item()
                total_loss_axes_angle += loss_axes_angle.item()
                total_error_up += np.mean(error_up)
                total_error_forward += np.mean(error_forward)
                max_error_up = max(max_error_up, abs(np.max(error_up)))
                max_error_forward = max(max_error_forward, abs(np.max(error_forward)))
                # total_loss_seg += loss_seg.item()
                count += 1
            print('  Testing loss: {}'.format(total_loss / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_axis_up', total_loss_axis_up / count, i)
            writer.add_scalar('test/loss_axis_forward', total_loss_axis_forward / count, i)
            writer.add_scalar('test/loss_axes_angle', total_loss_axes_angle / count, i)
            writer.add_scalar('test/loss_seg', total_loss_seg / count, i)
            writer.add_scalar('test/angle_error_up', total_error_up / count, i)
            writer.add_scalar('test/angle_error_forward', total_error_forward / count, i)
            writer.add_scalar('test/max_error_up', max_error_up, i)
            writer.add_scalar('test/max_error_forward', max_error_forward, i)
            writer.flush()

        if i % 100 == 0:
            lr = lr * 0.5
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=0, amsgrad=True
            )

        if i % 20 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.module.state_dict(),
                       os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_{}'.format(i))))

        if best_loss > total_loss / count:
            best_loss = total_loss / count
            torch.save(model.module.state_dict(),
                       os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_best')))
