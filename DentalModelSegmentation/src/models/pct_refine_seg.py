import os

import torch
from tqdm import tqdm

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors

import src.metrics.metrics as metrics


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



def model_func(patches, labels):
    patches = patches.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    seg_mask = torch.zeros(labels.size(), dtype=torch.int64).cuda()
    seg_mask[labels > 0] = 1

    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 8 + labels[
        labels > 0] % 10

    seg = model(patches.transpose(2, 1).contiguous())

    loss_seg = criterion(seg, seg_mask)
    #loss_seg = soft_dice_loss(seg_mask, torch.argmax(seg, 1))

    seg_int_tensor = torch.argmax(seg, 1)
    seg_int_tensor = seg_int_tensor.to(dtype=torch.int64).cuda()

    acc = metrics.mean_intersection_over_union(seg_int_tensor, seg_mask)

    return seg, loss_seg, acc


if __name__ == '__main__':
    from src.data.TeethPatchTrainingDataset import TeethPatchDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    from pct_models import *

    # experiment_dir = '/IGIP/zsj/projects/miccai-3d-teeth-seg/output/experiment_refine_pct_0730/'
    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_refine_pct_0801/'

    # data_dir = '/IGIP/zsj/data/miccai/h5_patches/'
    data_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches/'
    # train_set = TeethPatchDataset(data_dir, train=True)
    test_set = TeethPatchDataset(data_dir, train=False)
    train_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('Dataset ok')

    model = PctPatchRefine()
    model = torch.nn.DataParallel(model)
    model.cuda()

    torch.manual_seed(20000228)
    torch.cuda.manual_seed(20000228)

    lr = 1e-3
    optimizer_seg = optim.Adam(
        model.parameters(), lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    colors = AnatomyColors()
    best_acc = 0

    writer = TensorboardUtils(experiment_dir).writer

    for i in range(0, 500):
        total_loss = 0
        total_loss_seg = 0
        total_acc = 0
        count = 0

        model.train()
        with tqdm(total=len(train_loader)) as t:
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

        #print('Epoch {} - loss: {}, acc: {}'.format(i, total_loss / count, total_acc / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_seg = 0
            total_acc = 0
            count = 0
            with tqdm(total=len(test_loader)) as t:
                t.set_description('-- testing')
                for batch_i, batch_data in enumerate(test_loader):
                    patches, labels, centroids = batch_data

                    seg, loss, acc = model_func(patches, labels)
                    seg = torch.argmax(seg.squeeze(), 1)

                    if i % 3 == 0 and batch_i < 30:
                        for ii in range(0, len(patches), 4):
                            nd_data = patches[ii].data.cpu().numpy()
                            nd_seg = seg[ii].data.cpu().numpy()

                            nd_centroid = np.concatenate((centroids[ii, :, 0:3], np.array([[0, 0, 0, 1, 0, 0, 1]])), axis=1)
                            color_seg = np.zeros((nd_data.shape[0], 4)) * 0.3
                            color_gt = np.zeros((nd_data.shape[0], 4))
                            for pt in range(nd_data.shape[0]):
                                # color_seg[pt, 0:3] = colors.get_color(nd_cls, True) if nd_seg[pt] > 0.5 else np.ones((3, )) * 0.5
                                color_seg[pt, 0:3] = np.array([nd_seg[pt], nd_seg[pt], nd_seg[pt]])
                                color_seg[pt, 3] = 1
                                color_gt[pt, 0:3] = colors.get_color(labels[ii][pt].data.cpu().numpy(), True)
                                color_gt[pt, 3] = 1

                            np.savetxt(
                                '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_refine_pct/seg_{}_{}.txt'.format(batch_i, ii),
                                np.concatenate((np.concatenate((nd_data[:, 0:6], color_seg), axis=1), nd_centroid), axis=0))
                            np.savetxt(
                                '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_refine_pct/gt_{}.txt'.format(ii),
                                np.concatenate((nd_data[:, 0:6], color_gt), axis=1))

                    total_loss += loss.item()
                    total_acc += acc
                    count += 1
                    t.set_postfix(acc=acc, total_acc=total_acc / count)
                    t.update()
                #print('  |-- test loss: {}, acc: {}'.format(total_loss / count, total_acc / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/acc', total_acc / count, i)

        if i % 100 == 0:
            lr = lr * 0.5
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=0, amsgrad=True
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


