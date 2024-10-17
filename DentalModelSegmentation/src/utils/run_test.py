import os

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
import time
import numpy as np
from src.utils.postprocessing import fast_cluster

from src.utils.tensorboard_utils import TensorboardUtils


def output_color_point_cloud_path(data, preds_kp, seed, kp_score, out_file):
    with open(out_file, 'w') as f:
        l = data.shape[0]
        for i in range(seed.shape[0]):
            color = [0, 1, 0]
            f.write('v %f %f %f %f %f %f\n' % (seed[i][0], seed[i][1], seed[i][2], color[0], color[1], color[2]))
            color = [1, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (
            preds_kp[i][0], preds_kp[i][1], preds_kp[i][2], color[0], color[1], color[2]))
        for i in range(l):
            color = [0, 0, 1]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

        # the path of predicted centroid
        f.write('g lines\n')
        for i in range(seed.shape[0]):
            f.write('l %d %d\n' % (2 * i + 1, 2 * (i + 1)))


def output_color_point_cloud_red_blue(data, kp, kp_gt, out_file):
    with open(out_file, 'w') as f:
        l = data.shape[0]
        for i in range(data.shape[0]):
            color = [0, 0, 1]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
        for i in range(kp.shape[0]):
            color = [1, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (kp[i][0], kp[i][1], kp[i][2], color[0], color[1], color[2]))
        for i in range(kp_gt.shape[0]):
            color = [0, 1, 0]
            f.write('v %f %f %f %f %f %f\n' % (kp_gt[i][0], kp_gt[i][1], kp_gt[i][2], color[0], color[1], color[2]))


def test_stage1(epoch: int,
                model: nn.Module,
                dataloader: DataLoader):
    """
    Run test once

    Run test once on test set, and returns test loss, test accuracy.
    """
    # ------------- Init and print log -------------
    start_time = time.time()
    writer = TensorboardUtils().writer

    # ------------- Var definitions -------------
    mean_loss_test = 0
    mean_dist1_test = 0
    mean_dist2_test = 0
    mean_dist1_max_test = 0
    mean_dist2_max_test = 0
    mean_acc_test = 0

    count_test = 0

    # ------------- RUN -------------
    d1 = []
    d2 = []
    d1_std = []
    d2_std = []
    for i_batch, batch_data in enumerate(dataloader):
        model.eval()
        data = batch_data[0]
        labels = batch_data[1]
        centroids = batch_data[2]

        data = data.to("cuda", dtype=torch.float32, non_blocking=True)
        labels = labels.to("cuda", dtype=torch.int32, non_blocking=True)
        centroids = centroids.to("cuda", dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            kp_reg, kp_score, seed_xyz, loss, dist1, dist2, dist1_max, dist2_max = model(data, centroids, True)

            mean_loss_test = mean_loss_test + loss.item()
            mean_dist1_test = mean_dist1_test + dist1
            mean_dist2_test = mean_dist2_test + dist2
            mean_dist1_max_test = mean_dist1_max_test + dist1_max
            mean_dist2_max_test = mean_dist2_max_test + dist2_max
            count_test = count_test + 1
            # mean_acc_test += metrics.teeth_localization_accuracy(kp_reg.detach().cpu(), centroids.detach().cpu())

            # test
            d1.append(dist1.item())
            d2.append(dist2.item())
            d1_std.append(dist1_max.item())
            d2_std.append(dist2_max.item())

            if epoch % 3 == 0:
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
                    for bid in range(5):
                        np.savetxt(os.path.join(snapshot_path + 'validate/cent' + str(count_test) + '.xyz'), preds_kp[bid, kp_score[bid, :] < 0.2])
                        output_color_point_cloud_path(data[bid, :, :], preds_kp[bid, kp_score[bid, :] < 0.2],
                                                      seed_xyz[bid, kp_score[bid, :] < 0.2], kp_score[bid, :],
                                                      os.path.join(
                                                          snapshot_path + f'validate/sample{count_test}_{bid}.obj'))
                        np.savetxt(os.path.join(snapshot_path + f'validate/gt{count_test}_{bid}.txt'), centroids[bid, :, :].detach().cpu().numpy())

    writer.add_scalar('testing/loss', mean_loss_test / count_test, epoch)
    writer.add_scalar('testing/dist1', mean_dist1_test / count_test, epoch)
    writer.add_scalar('testing/dist2', mean_dist2_test / count_test, epoch)
    writer.add_scalar('testing/dist1_max', mean_dist1_max_test / count_test, epoch)
    writer.add_scalar('testing/dist2_max', mean_dist2_max_test / count_test, epoch)

    mean_loss_test = mean_loss_test/count_test
    logger.info('Test spent {}s, loss: {}', time.time() - start_time, mean_loss_test)
    return mean_loss_test
