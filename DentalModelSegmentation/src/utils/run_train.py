import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from src.pointnet2.pointnet2_utils import FurthestPointSampling
from torch import optim
import time
import src.utils.run_test as run_test

from src.models import *
from src.utils.postprocessing import fast_cluster, sample_to_points_cuda
from src.utils.tensorboard_utils import TensorboardUtils


def train_one_epoch_stage1(epoch: int,
                           model,
                           dataloaders: list,
                           optimizer,
                           weight_decay: float = 0,
                           lr: float = 1e-3,
                           lr_decay: float = 0.5,
                           decay_step: int = 500):
    """
    Run a training epoch: centroid prediction

    Run a full training epoch for all data in training set. The procedure includes
    forward propagation, loss computation, backward propagation, calculating accuracy
    on test set.
    """
    # ------------- Init and print log -------------
    logger.info('========== Stage 1, Epoch {} ==========', epoch)
    start_time = time.time()
    writer = TensorboardUtils().writer

    # ------------- Var definitions -------------
    mean_loss = 0
    count_train = 0
    train_loader, test_loader = dataloaders

    mean_dist1 = 0
    mean_dist2 = 0
    mean_dist1_max = 0
    mean_dist2_max = 0
    mean_acc = 0

    # ------------- RUN -------------
    for i_batch, batch_data in enumerate(train_loader):
        torch.cuda.empty_cache()
        model.train()
        data = batch_data[0]
        centroids = batch_data[2]

        data = data.to("cuda", dtype=torch.float32, non_blocking=True)
        centroids = centroids.to("cuda", dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()
        kp_reg, kp_score, seed_xyz, loss, dist1, dist2, dist1_max, dist2_max = model(data, centroids, True)

        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            if count_train < 40:
                data = data.data.cpu().numpy()[:, :, :3]
                preds_kp = kp_reg.data.cpu().numpy()
                # score_reg = score_reg.data.cpu().numpy()
                # score_gt = score_gt.data.cpu().numpy()
                seed_xyz = seed_xyz.data.cpu().numpy()
                kp_score = kp_score.data.cpu().numpy()
                # output_color_point_cloud_red_blue(data[0, :, :], preds_kp[0, score_reg[0, :] > 0.5], centroids,
                #                                   os.path.join('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/' + str(epoch) + '_up_conf_' + str(count_test) + '.obj'))
                snapshot_path = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/'
                if not os.path.exists(snapshot_path + 'validate/'):
                    os.makedirs(snapshot_path + 'validate/')
                # out_kp = fast_cluster(preds_kp[0, kp_score[0, :] < 0.2])
                # np.savetxt(os.path.join(snapshot_path + 'validate/gt' + str(count_test) + '.txt'), out_kp)
                run_test.output_color_point_cloud_path(data[0, :, :], preds_kp[0, kp_score[0, :] < 0.2],
                                                       seed_xyz[0, kp_score[0, :] < 0.2], kp_score[0, :],
                                                       os.path.join(
                                                  snapshot_path + 'validate/train' + str(count_train) + '_0.obj'))
                run_test.output_color_point_cloud_path(data[1, :, :], preds_kp[1, kp_score[1, :] < 0.2],
                                                       seed_xyz[1, kp_score[1, :] < 0.2], kp_score[1, :],
                                                       os.path.join(
                                                  snapshot_path + 'validate/train' + str(count_train) + '_1.obj'))
                # np.savetxt(os.path.join(snapshot_path + 'validate/gt' + str(count_test) + '.txt'), centroids[0, :, :].detach().cpu().numpy())

        # ------------- Clear GPU memory -------------
        kp_reg = kp_reg.detach().cpu()
        # kp_score = kp_score.detach().cpu()
        # seed_xyz = seed_xyz.detach().cpu()
        loss = loss.detach().cpu()
        dist1 = dist1.detach().cpu()
        dist2 = dist2.detach().cpu()
        dist1_max = dist1_max.detach().cpu()
        dist2_max = dist2_max.detach().cpu()
        # data = data.detach().cpu()
        centroids = centroids.detach().cpu()
        torch.cuda.empty_cache()
        # ------------- End clear GPU memory -------------

        mean_loss = mean_loss + loss.item()
        mean_dist1 = mean_dist1 + dist1
        mean_dist2 = mean_dist2 + dist2
        mean_dist1_max = mean_dist1_max + dist1_max
        mean_dist2_max = mean_dist2_max + dist2_max
        count_train = count_train + 1

    writer.add_scalar('training/loss', mean_loss / count_train, epoch)
    writer.add_scalar('training/dist1', mean_dist1 / count_train, epoch)
    writer.add_scalar('training/dist2', mean_dist2 / count_train, epoch)
    writer.add_scalar('training/dist1_max', mean_dist1_max / count_train, epoch)
    writer.add_scalar('training/dist2_max', mean_dist2_max / count_train, epoch)
    # writer.add_scalar('training/TLA_acc', mean_acc / count_train, epoch)

    # ------------- LR decay -------------
    if epoch % decay_step == 0 and epoch > 0:
        optimizer = optim.Adam(
            model.parameters(), lr=lr * lr_decay, weight_decay=weight_decay,
            amsgrad=True
        )
        logger.info('Learning rate decay: [{}] -> [{}]', lr, lr * lr_decay)
        lr = lr * lr_decay

    mean_loss = mean_loss / count_train

    logger.info('Training spent {}s, loss: {}, acc: {}', time.time() - start_time, mean_loss, mean_acc / count_train)

    # ------------- Test -------------
    mean_loss_test = run_test.test_stage1(epoch, model, test_loader)

    writer.flush()
    # ------------- Update checkpoint data -------------
    return optimizer, lr, mean_loss_test


def train_one_epoch_stage2(epoch: int,
                           models: list,
                           model_centroid_prediction: PointnetCentroidPredictionNet,
                           dataloaders: list,
                           optimizer,
                           weight_decay: float = 0,
                           lr: float = 1e-3,
                           lr_decay: float = 0.5,
                           decay_step: int = 500):
    """
    Run a training epoch: tooth segmentation

    Run a full training epoch for all data in training set. The procedure includes
    forward propagation, loss computation, backward propagation, calculating accuracy
    on test set.
    """
    # ------------- Init and print log -------------
    logger.info('========== Stage 2, Epoch {} ==========', epoch)
    start_time = time.time()
    writer = TensorboardUtils().writer

    # ------------- Var definitions -------------
    mean_loss = 0
    count_train = 0
    train_loader, test_loader = dataloaders

    mean_acc = 0

    model_seg, model_seg_refine = models

    for i_batch, batch_data in enumerate(train_loader):
        torch.cuda.empty_cache()
        model_seg.train()
        model_seg_refine.train()

        data = batch_data[0]
        labels = batch_data[1]
        centroids = batch_data[2]

        data = data.to("cuda", dtype=torch.float32, non_blocking=True)
        labels = labels.to("cuda", dtype=torch.float32, non_blocking=True)
        centroids = centroids.to("cuda", dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            # Predict and cluster centroids
            pred_centroids, conf_centroids, seed_xyz = model_centroid_prediction(data, False, False)
            # pred_centroids = pred_centroids.detach().cpu().numpy()
            # conf_centroids = conf_centroids.detach().cpu().numpy()

        # Total loss
        batch_mean_loss = 0
        count_small_batch = 0
        for batch_id in range(len(data)):
            data_cropped_indices, data_cropped_list, labels_cropped_list = [], [], []
            pred_centroids_one_batch = pred_centroids[batch_id, conf_centroids[batch_id, :] > 0.3, :]
            # FPS centroids
            if len(pred_centroids_one_batch) > 16:
                centroid_fps_indices = FurthestPointSampling.apply(pred_centroids_one_batch[:, :3].unsqueeze(0), 16) \
                    .type(torch.LongTensor)
                centroid_fps_indices = centroid_fps_indices[0]
            else:
                centroid_fps_indices = torch.arange(0, len(pred_centroids_one_batch))

            np.savetxt('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/cluster_origin.xyz', pred_centroids_one_batch.data.cpu().numpy())
            pred_centroids_one_batch = pred_centroids_one_batch[centroid_fps_indices, :].detach().cpu().numpy()

            # pred_centroids = fast_cluster(pred_centroids)

            # Crop patches according to predicted centroids
            crop_size = len(data[batch_id]) // 4

            # Perform crop
            nd_data = data.data.cpu().numpy()
            nd_labels = labels.data.cpu().numpy()

            for crop_id in range(pred_centroids_one_batch.shape[0]):
                data_points_cropped_dis = \
                    np.sqrt(np.sum((nd_data[batch_id, :, :3] - pred_centroids_one_batch[crop_id, :]) ** 2, axis=1))
                data_points_cropped_index = np.argpartition(data_points_cropped_dis, crop_size)[:crop_size]
                data_points_cropped = nd_data[batch_id, data_points_cropped_index, :]
                labels_cropped = nd_labels[batch_id, data_points_cropped_index]

                # Normalize
                min_p = data_points_cropped[:, :3].min(axis=0)
                data_points_cropped[:, :3] = data_points_cropped[:, :3] - min_p
                scale = np.max(np.sqrt(np.sum(data_points_cropped[:, :3] ** 2, axis=1)))
                data_points_cropped[:, :3] = data_points_cropped[:, :3] / scale
                pred_centroids_trans = (pred_centroids_one_batch[crop_id, :] - min_p) / scale
                data_points_cropped[:, :3] = data_points_cropped[:, :3] - pred_centroids_trans

                points_dis = np.sqrt(np.sum(data_points_cropped[:, :3] ** 2, axis=1))
                heatmap = np.exp(-4 * points_dis)
                data_points_cropped = np.concatenate((data_points_cropped[:, :6],
                                                      heatmap.reshape((heatmap.shape[0], 1))), axis=1)

                data_cropped_indices.append(data_points_cropped_index)
                data_cropped_list.append(data_points_cropped)
                labels_cropped_list.append(labels_cropped)

            # len(data_cropped_list) is too large, split it
            small_batch_size = 16
            for small_batch_id in range(0, len(data_cropped_list), small_batch_size):
                # ------------- RUN -------------
                optimizer.zero_grad()
                # Choose this batch and next batch's data, so batch_size is at least 2,
                # or batch_norm will throw an exception
                data_points_cropped = torch.from_numpy(np.asarray(
                    data_cropped_list[small_batch_id:small_batch_id + small_batch_size])).type(torch.FloatTensor)
                labels_cropped = torch.Tensor(
                    np.array(labels_cropped_list[small_batch_id:(small_batch_id + small_batch_size)])).type(torch.FloatTensor)

                data_points_cropped = data_points_cropped.to("cuda", non_blocking=True)
                labels_cropped = labels_cropped.to("cuda", non_blocking=True)

                # FPS
                data_pts_idx = FurthestPointSampling.apply(data_points_cropped[:, :, :3].contiguous(), 4096) \
                    .type(torch.LongTensor)
                sampled_pts_idx_viewed = data_pts_idx.view(data_pts_idx.shape[0] * data_pts_idx.shape[1]).cuda().type(
                    torch.LongTensor)
                batch_idxs = torch.tensor(range(data_points_cropped.shape[0])).type(torch.LongTensor)
                batch_idxs_viewed = batch_idxs[:, None].repeat(1, data_pts_idx.shape[1]).view(
                    batch_idxs.shape[0] * data_pts_idx.shape[1])
                sampled_pts = data_points_cropped[batch_idxs_viewed, sampled_pts_idx_viewed, :]
                sampled_labels = labels_cropped[batch_idxs_viewed, sampled_pts_idx_viewed] \
                    .view(labels_cropped.shape[0], 4096)
                sampled_labels[sampled_labels > 1] = 1

                # FPS result
                data_points_cropped_samp = sampled_pts.view(data_points_cropped.shape[0], 4096, 7)
                if data_points_cropped_samp.shape[0] == 1:
                    continue

                # First segmentation network
                preds_seg, preds_conf, preds_cls = model_seg(data_points_cropped_samp)
                loss_s1 = torch.mean((nn.BCELoss(reduction='none')(preds_seg, sampled_labels) * preds_conf) ** 2
                                     + (1 - preds_conf) ** 2)

                preds_seg = torch.unsqueeze(preds_seg.detach(), 2)
                data_points_cropped_samp = torch.cat((data_points_cropped_samp, preds_seg), 2)

                # Second segmentation refine network
                preds_seg, preds_conf, preds_cls = model_seg_refine(data_points_cropped_samp)
                preds_seg_data = sample_to_points_cuda(torch.unsqueeze(preds_seg[:, :], 1),
                                                       data_points_cropped_samp[:, :, :3],
                                                       data_points_cropped[:, :, :3])

                preds_conf = sample_to_points_cuda(torch.unsqueeze(preds_conf[:, :], 1),
                                                   data_points_cropped_samp[:, :, :3],
                                                   data_points_cropped[:, :, :3])

                labels_cropped[labels_cropped > 0] = 1
                loss_s2 = torch.mean((2 - preds_conf) * nn.BCELoss(reduction='none')(preds_seg_data, labels_cropped))

                loss_seg = loss_s1 + loss_s2

                loss_seg.backward()
                optimizer.step()

                batch_mean_loss += loss_seg.item()
                count_small_batch += 1
        batch_mean_loss /= count_small_batch

        # ------------- Clear GPU memory -------------
        data = data.detach().cpu()
        centroids = centroids.detach().cpu()
        torch.cuda.empty_cache()
        # ------------- End clear GPU memory -------------

        mean_loss = mean_loss + batch_mean_loss
        count_train = count_train + 1

    writer.add_scalar('training/loss', mean_loss / count_train, epoch)

    # ------------- LR decay -------------
    if epoch % decay_step == 0 and epoch > 0:
        optimizer = optim.Adam(
            model_seg.parameters(), lr=lr * lr_decay, weight_decay=weight_decay
        )
        logger.info('Learning rate decay: [{}] -> [{}]', lr, lr * lr_decay)

    mean_loss = mean_loss / count_train

    logger.info('Training spent {}s, loss: {}, acc: {}', time.time() - start_time, mean_loss, mean_acc / count_train)

    # ------------- Test -------------
    # run_test.test_stage1(epoch, model, test_loader)

    # ------------- Update checkpoint data -------------
    return optimizer, lr, epoch % 100 == 0
