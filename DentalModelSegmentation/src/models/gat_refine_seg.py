from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch
import tqdm

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors

import src.metrics.metrics as metrics


def model_func(patches, labels):
    patches = patches.to("cuda", dtype=torch.float32, non_blocking=True)
    seg_mask = torch.zeros(labels.size(), dtype=torch.int64).cuda()
    seg_mask[labels > 0] = 1

    seg = model(patches.transpose(2, 1).contiguous())
    seg = seg.permute(0, 2, 1)

    seg_int_tensor = torch.argmax(seg, -1)
    seg_int_tensor = seg_int_tensor.to(dtype=torch.int32).cuda()
    acc = metrics.mean_intersection_over_union(seg_int_tensor, seg_mask)

    loss_seg = F.cross_entropy(seg.permute(0, 2, 1), seg_mask, reduction='mean')

    return seg, loss_seg, acc


if __name__ == '__main__':
    from src.data.TeethPatchTrainingDataset import TeethPatchDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    from dgcnn_models import *

    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_refine_dgcnn_gat/'
    writer = TensorboardUtils(experiment_dir).writer

    data_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches_small/'
    train_set = TeethPatchDataset(data_dir, train=True)
    # test_set = TeethPatchDataset(data_dir, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('Dataset ok')

    model = DGCNN_attention_semseg({
        'k': 20,
        'emb_dims': 1024,
        'dropout': 0.5
    })
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_refine_dgcnn/snapshots/model_200'))
    model.cuda()

    lr = 1e-4
    optimizer_seg = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=0, amsgrad=True
    )

    criterion = nn.CrossEntropyLoss()

    colors = AnatomyColors()
    best_acc = 0

    for i in range(0, 1000):
        total_loss = 0
        total_loss_seg = 0
        total_acc = 0
        count = 0

        model.train()
        for batch_i, batch_data in enumerate(tqdm.tqdm(train_loader)):
            optimizer_seg.zero_grad()
            # Let "P" be the amount of patches
            #   patches: [P, 4096, 6], P patches in total, each containing 4096 points
            #   labels: [P, 4096], ground truth label for each point
            #   indices: [P, 4096], selected from original teeth model point cloud that formed the patches
            #   seg: [P, 4096], predicted segmentation confidence
            #   cls: [P, 4096], predicted classification labels
            #   filename: str, filename of this model
            patches, labels, _ = batch_data

            _, loss, acc = model_func(patches, labels)

            loss.backward()
            optimizer_seg.step()

            loss.detach()

            total_loss += loss.item()
            total_acc += acc
            count += 1

        print('Epoch {} - loss: {}'.format(i, total_loss / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        writer.add_scalar('train/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_seg = 0
            total_acc = 0
            count = 0
            for batch_i, batch_data in enumerate(tqdm.tqdm(test_loader)):
                patches, labels, centroids = batch_data

                seg, loss, acc = model_func(patches, labels)
                seg = torch.argmax(seg.squeeze(), -1)

                if batch_i < 10000:
                    for ii in range(0, len(patches), 4):
                        nd_data = patches[ii].data.cpu().numpy()
                        nd_seg = seg[ii].data.cpu().numpy()

                        nd_centroid = np.concatenate((centroids[ii, :, 0:3], np.array([[0, 0, 0, 1, 0, 0, 1]])), axis=1)
                        color_seg = np.zeros((nd_data.shape[0], 4))
                        color_gt = np.zeros((nd_data.shape[0], 4))
                        for pt in range(nd_data.shape[0]):
                            # color_seg[pt, 0:3] = colors.get_color(nd_cls, True) if nd_seg[pt] > 0.5 else np.ones((3, )) * 0.5
                            color_seg[pt, 0:3] = np.array([nd_seg[pt], nd_seg[pt], nd_seg[pt]])
                            color_seg[pt, 3] = 1
                            color_gt[pt, 0:3] = colors.get_color(labels[ii][pt], True)
                            color_gt[pt, 3] = 1

                        np.savetxt(
                            '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_refine_gat/seg_{}_{}.txt'.format(batch_i, ii),
                            np.concatenate((np.concatenate((nd_data[:, 0:6], color_seg), axis=1), nd_centroid), axis=0))
                        # np.savetxt(
                        #     '/IGIP/zsj/projects/miccai-3d-teeth-seg/temp/validate/test_refine/gt_{}.txt'.format(ii),
                        #     np.concatenate((nd_data[:, 0:6], color_gt), axis=1))

                total_loss += loss.item()
                total_acc += acc
                count += 1
            print('  |-- test loss: {}'.format(total_loss / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_seg', total_loss_seg / count, i)
            writer.add_scalar('test/acc', total_acc / count, i)

        if i % 50 == 0:
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

