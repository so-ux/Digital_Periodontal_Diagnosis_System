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

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors

import src.metrics.metrics as metrics
import os


_eps = 1e-12


def boundary_mining_without_batch(p1, p2, gt, radii):
    """
    :param gt: size[N]
    :param radii:
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    """

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=2)
    # print(dist)
    dist_sort, indices = torch.sort(dist)
    indices = indices[:, 1:33]

    gt_gt = gt.unsqueeze(0).repeat(gt.shape[0], 1)
    gt_gt = torch.gather(gt_gt, 0, indices)
    gt = gt.unsqueeze(1).repeat(1, 32)
    # for i in range(len(gt_gt)):
    #     gt_gt[i] = gt_gt[i][indices[i]]
    # gt_gt = gt_gt[indices]
    # 在radii球内的点mask
    rad_mask = dist_sort[:, 1:33] > radii
    pos_mask = gt_gt == gt
    neg_mask = gt_gt != gt

    pos_mask[rad_mask] = False
    # for i in range(p1.shape[0]):
    #     pos_mask[i, i] = False

    neg_mask[rad_mask] = False
    pos_cnt = torch.sum(pos_mask, dim=-1)
    neg_cnt = torch.sum(neg_mask, dim=-1)
    boundary = (pos_cnt * neg_cnt) > 0
    return boundary, pos_mask, neg_mask, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_semseg(nn.Module):
    def __init__(self, k=32):
        super(DGCNN_semseg, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(0.1)
        self.conv9 = nn.Conv1d(256, 2, kernel_size=1, bias=False)

    def boundary_mask(self, xyz, gt, rad=0.1):
        assert len(xyz.shape) == 3 and xyz.shape[2] == 3
        assert len(gt.shape) == 2 and gt.shape[0] == xyz.shape[0] and gt.shape[1] == xyz.shape[1]

        b_masks, pos_masks, neg_masks, indices = [], [], [], []
        for bid in range(gt.shape[0]):
            boundary_mask, pos_mask, neg_mask, idx = boundary_mining_without_batch(xyz[bid].unsqueeze(0), xyz[bid].unsqueeze(0), gt[bid], rad)
            b_masks.append(boundary_mask)
            pos_masks.append(pos_mask)
            neg_masks.append(neg_mask)
            indices.append(idx)
        b_masks = torch.stack(b_masks)
        pos_masks = torch.stack(pos_masks)
        neg_masks = torch.stack(neg_masks)
        indices = torch.stack(indices)
        return b_masks, pos_masks, neg_masks, indices

    def contrastive_loss(self, dist, posmask, temperature=1):
        print('cont loss', dist.shape, posmask.shape)
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        dist = dist / temperature
        exp = torch.exp(dist)
        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        return loss

    def cal_dist(self, a, b, feats, mask):
        return torch.sum(feats[mask] * feats, dim=-1) / (torch.norm(feats[mask], dim=-1) * torch.norm(feats, dim=-1))

    def dist_l2(self, features, indices, b_masks, pos_masks, neg_masks):
        features = features.permute(0, 2, 1)
        B, N, C = features.shape
        losses = []
        tau = 1
        for bi in range(B):
            loss_b = []
            # pos_neighbours = pos_masks[bi].to(dtype=torch.int64)
            # neg_neighbours = neg_masks[bi].to(dtype=torch.int64)
            # feat = features[bi].unsqueeze(0)
            # pos_dist = self.cal_dist(features[bi].gather(1, pos_neighbours), features[bi])
            # neg_dist = self.cal_dist(features[bi].gather(1, neg_neighbours), features[bi])
            # exp = torch.sum(torch.exp(-pos_dist / tau), dim=-1)
            # exp_all = torch.sum(torch.exp(-neg_dist / tau), dim=-1) + exp
            # losses.append(-1 * torch.mean(torch.log(exp / exp_all)))

            feat = features[bi].unsqueeze(1)
            nn_feats = features[bi][indices[bi]]
            f_dist = torch.sum((nn_feats - feat) ** 2, dim=-1)
            print(f_dist.shape, pos_masks.shape)
            f_dist[~b_masks[bi]] = 0
            print(f_dist.shape, pos_masks.shape)
            pos_dist = torch.sum(f_dist[pos_masks[bi]], dim=-1)
            neg_dist = torch.sum(f_dist[neg_masks[bi]], dim=-1)
            # pos_dist = torch.gather(f_dist, 0, pos_masks[bi])
            # neg_dist = torch.gather(f_dist, 0, neg_masks[bi])
            exp = torch.sum(torch.exp(-pos_dist / tau))
            exp_all = torch.sum(torch.exp(-neg_dist / tau)) + exp
            loss_b.append(torch.log(exp / exp_all))

            # for ni in range(N):
            #     if not b_masks[bi, ni]:
            #         continue
            #     pos_neighbours = pos_masks[bi, ni, :]
            #     neg_neighbours = neg_masks[bi, ni, :]
            #     feat = features[bi, ni, :].unsqueeze(0)
            #
            #     pos_dist = torch.sum(features[bi][pos_neighbours] * feat, dim=-1) / (torch.norm(feat) * torch.norm(features[bi][pos_neighbours], dim=-1))
            #     neg_dist = torch.sum(features[bi][neg_neighbours] * feat, dim=-1) / (torch.norm(feat) * torch.norm(features[bi][neg_neighbours], dim=-1))
            #
            #     exp = torch.sum(torch.exp(-pos_dist / tau))
            #     exp_all = torch.sum(torch.exp(-neg_dist / tau)) + exp
            #     loss_b.append(torch.log(exp / exp_all))
            loss_b = torch.stack(loss_b)
            losses.append(-1 * torch.mean(loss_b))
        losses = torch.stack(losses)
        return torch.mean(losses)

    def forward(self, x, gt=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x[:, 0:3, :].permute(0, 2, 1)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        if gt is not None:
            b_masks, pos_masks, neg_masks, indices = self.boundary_mask(xyz, gt)
            # for bid in range(len(b_masks)):
            #     ndcolor = np.array([[1, 1, 1, 0.5, 0.5, 0.5, 1]])
            #     ndcolor = np.repeat(ndcolor, repeats=4096, axis=0)
            #     ndcolor[b_masks[bid].data.cpu().numpy(), 3] = 1
            #     ndout = np.concatenate((xyz[bid].data.cpu().numpy(), ndcolor), axis=-1)
            #
            #     np.savetxt(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches_nonuniform/vis{bid}.txt', ndout)
            # exit(0)
            # 计算样本间的距离矩阵
            loss_cbl = self.dist_l2(x, indices, b_masks, pos_masks, neg_masks)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        if gt is not None:
            return torch.softmax(x, dim=1), loss_cbl
        else:
            return torch.softmax(x, dim=1)


def model_func(patches, labels):
    patches = patches.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    seg_mask = torch.zeros(labels.size(), dtype=torch.int64).cuda()
    seg_mask[labels > 0] = 1

    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 8 + labels[
        labels > 0] % 10
    label_1d = torch.max(labels, dim=-1)[0]

    seg, loss_cbl = model(patches.transpose(2, 1).contiguous(), seg_mask)

    loss_seg = criterion(seg, seg_mask)

    # seg_int_tensor = torch.zeros(seg.size())
    # seg_int_tensor[seg > 0.5] = 1
    seg_int_tensor = torch.argmax(seg, dim=1)
    seg_int_tensor = seg_int_tensor.to(dtype=torch.int32).cuda()
    acc = metrics.mean_intersection_over_union(seg_int_tensor, seg_mask)

    return seg, loss_seg + 0.1 * loss_cbl, acc


if __name__ == '__main__':
    from src.data.TeethPatchCBLTrainingDataset import TeethPatchDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np

    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_refine_dgcnn_cbl_0827/'
    writer = TensorboardUtils(experiment_dir).writer

    data_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches_nonuniform/'
    train_set = TeethPatchDataset(data_dir, train=True)
    # test_set = TeethPatchDataset(data_dir, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('Dataset ok')

    model = DGCNN_semseg()
    model = torch.nn.DataParallel(model)
    model.cuda()

    lr = 1e-3
    optimizer_seg = optim.Adam(
        model.parameters(), lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    colors = AnatomyColors()
    best_acc = 0

    for i in range(0, 600):
        total_loss = 0
        total_loss_seg = 0
        total_acc = 0
        count = 0

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as t:
            t.set_description('Epoch %i' % i)
            for batch_i, batch_data in enumerate(train_loader):
                optimizer_seg.zero_grad()
                patches, labels, _ = batch_data

                _, loss, acc = model_func(patches, labels)

                loss.backward()
                optimizer_seg.step()

                loss.detach()

                total_loss += loss.item()
                total_acc += acc
                count += 1
                t.set_postfix(acc=acc, total_acc=total_acc / count)
                t.update()

        # print('Epoch {} - loss: {}'.format(i, total_loss / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        writer.add_scalar('train/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_seg = 0
            total_acc = 0
            count = 0

            with tqdm.tqdm(total=len(train_loader)) as t:
                t.set_description('Epoch %i' % i)
                for batch_i, batch_data in enumerate(test_loader):
                    patches, labels, centroids = batch_data

                    seg, loss, acc = model_func(patches, labels)
                    seg = seg.squeeze()

                    if batch_i < 10:
                        for ii in range(0, len(patches), 4):
                            nd_data = patches[ii].data.cpu().numpy()
                            nd_seg = seg[ii].data.cpu().numpy()

                            nd_centroid = np.concatenate((centroids[ii, :, 0:3], np.array([[0, 0, 0, 1, 0, 0, 1]])), axis=1)
                            color_seg = np.zeros((nd_data.shape[0], 4))
                            #color_gt = np.zeros((nd_data.shape[0], 4))
                            for pt in range(nd_data.shape[0]):
                                # color_seg[pt, 0:3] = colors.get_color(nd_cls, True) if nd_seg[pt] > 0.5 else np.ones((3, )) * 0.5
                                color_seg[pt, 0:3] = np.array([nd_seg[pt], nd_seg[pt], nd_seg[pt]])
                                color_seg[pt, 3] = 1
                                #color_gt[pt, 0:3] = colors.get_color(labels[ii][pt], True)
                                #color_gt[pt, 3] = 1

                            np.savetxt(
                                '/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/seg_{}_{}.txt'.format(batch_i, ii),
                                np.concatenate((np.concatenate((nd_data[:, 0:6], color_seg), axis=1), nd_centroid), axis=0))
                            # np.savetxt(
                            #     '/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/gt_{}.txt'.format(ii),
                            #     np.concatenate((nd_data[:, 0:6], color_gt), axis=1))

                    total_loss += loss.item()
                    total_acc += acc
                    count += 1
                    t.set_postfix(acc=acc, total_acc=total_acc / count)
                    t.update()
            # print('  |-- test loss: {}'.format(total_loss / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_seg', total_loss_seg / count, i)
            writer.add_scalar('test/acc', total_acc / count, i)

        if i % 30 == 0:
            lr = lr * 0.8
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=1e-4
            )

        if i % 20 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.module.state_dict(),
                       os.path.join(experiment_dir,
                                    'snapshots', 'model_{}'.format(i)))

        if best_acc < total_acc / count:
            best_acc = total_acc / count
            torch.save(model.module.state_dict(),
                       os.path.join(experiment_dir,
                                    'snapshots', 'model_{}'.format(i)))
        writer.flush()

