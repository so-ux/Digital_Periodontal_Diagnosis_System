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
from src.models.pct_models import PctClass

from tqdm import tqdm

from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.anatomy_colors import AnatomyColors


class TeethClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_tooth = PctClass(7)
        self.whole_tooth = PctClass(7)
        # self.FC_layer_cls = (
        #     pt_seq.Seq.builder(512 * 2)
        #     .fc(512, bn=True)
        #     .dropout(0.2)
        #     .fc(33, activation=None)
        # )
        self.tooth_id = (
            pt_seq.Seq.builder(512 * 2 + 5)
            .fc(256, bn=True)
            .dropout(0.1)
            .fc(33, activation=None)
        )

        self.tooth_quadrant = (
            pt_seq.Seq.builder(512)
            .fc(128, bn=True)
            .dropout(0.1)
            .fc(5, activation=None)
        )
        # self.tooth_id = (
        #     pt_seq.Seq.builder(512 * 2)
        #     .fc(256, bn=True)
        #     .dropout(0.1)
        #     .fc(9, activation=None)
        # )
        #
        # self.tooth_quadrant = (
        #     pt_seq.Seq.builder(512)
        #     .fc(128, bn=True)
        #     .dropout(0.1)
        #     .fc(5, activation=None)
        # )

    def forward(self, points, resamples):
        f1 = self.single_tooth(resamples)
        f2 = self.whole_tooth(points)
        # tooth_id = torch.softmax(self.tooth_id(torch.cat((f1, f2), dim=-1)), dim=1)
        tooth_quadrant = torch.softmax(self.tooth_quadrant(f2), dim=1)
        f_id = torch.cat((f1, f2, tooth_quadrant), dim=1)
        return torch.softmax(self.tooth_id(f_id), dim=1), tooth_quadrant
        # pred_cls = tooth_quadrant * 10 + tooth_id
        # pred_cls[pred_cls < 11] = 0
        # pred_cls[pred_cls % 10 == 0] = 0
        # return pred_cls

def model_func(points, labels, masks):
    """

    :param points: [B, N, 7]
    :type points:
    :param labels: [B, 1]
    :type labels:
    :return:
    :rtype:
    """
    points = points.to("cuda", dtype=torch.float32, non_blocking=True)
    masks = masks.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)

    tooth_id, tooth_quadrant = model(points, masks)

    labels_33 = torch.clone(labels)
    labels_33[labels_33 > 0] = (torch.div(labels_33[labels_33 > 0], 10, rounding_mode='trunc') - 1) * 8 + labels_33[labels_33 > 0] % 10

    loss_cls = [criterion_id(tooth_id, labels_33), criterion_quad(tooth_quadrant, torch.div(labels, 10, rounding_mode='trunc'))]

    tooth_id_argmax = torch.argmax(tooth_id, dim=1)
    return tooth_id_argmax, loss_cls, acc(tooth_id_argmax, labels_33), acc(
        torch.argmax(tooth_quadrant, dim=1), torch.div(labels, 10, rounding_mode='trunc'))


def acc(pred_cls, labels):
    """

    :param pred_cls: [B, 33]
    :type pred_cls:
    :param labels: [B, 1]
    :type labels:
    :return:
    :rtype:
    """
    return torch.sum(pred_cls == labels.view(-1)) / pred_cls.shape[0]


if __name__ == '__main__':
    import os
    from src.data.TeethClassDataset import TeethClassDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np

    experiment_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/output/experiment_tooth_cls_level_0811/'
    writer = TensorboardUtils(experiment_dir).writer

    data_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_cls/'

    train_set = TeethClassDataset(data_dir, train=True, require_dicts=True)
    test_set = TeethClassDataset(data_dir, train=False, require_dicts=True)
    train_loader = DataLoader(
        train_set,
        batch_size=48,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        test_set,
        batch_size=48,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('Dataset ok')

    # model = Pointnet2MSG(4)
    model = TeethClassifier()
    model.cuda()

    lr = 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=0, amsgrad=True
    )

    colors = AnatomyColors()

    best_acc = 0

    weight = torch.Tensor([
        3,
        1, 1, 1, 1, 1, 1, 1, 3,
        1, 1, 1, 1, 1, 1, 1, 3,
        1, 1, 1, 1, 1, 1, 1, 3,
        1, 1, 1, 1, 1, 1, 1, 3
    ]).cuda()
    criterion_id = nn.CrossEntropyLoss(weight=weight)
    criterion_quad = nn.CrossEntropyLoss()


    for i in range(1000):
        total_loss = 0
        total_acc = 0
        total_quad_acc = 0
        total_loss_id = 0
        total_loss_quadrant = 0
        count = 0

        model.train()
        with tqdm(total=len(train_loader)) as t:
            t.set_description('Epoch %i' % i)
            for batch_i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                # Let "P" be the amount of patches
                #   patches: [P, 4096, 6], P patches in total, each containing 4096 points
                #   labels: [P, 4096], ground truth label for each point
                #   indices: [P, 4096], selected from original teeth model point cloud that formed the patches
                #   seg: [P, 4096], predicted segmentation confidence
                #   cls: [P, 4096], predicted classification labels
                #   filename: str, filename of this model
                patches, labels, masks, _ = batch_data

                cls, losses, cls_acc, quad_acc = model_func(patches, labels, masks)

                loss_cls = losses[0] + losses[1]

                loss_cls.backward()
                optimizer.step()

                total_loss += loss_cls.item()
                total_acc += cls_acc.item()
                total_quad_acc += quad_acc.item()
                total_loss_id += losses[0].item()
                total_loss_quadrant += losses[1].item()
                count += 1
                t.set_postfix(acc=cls_acc.item(), quad_acc=quad_acc.item(), total_acc=total_acc / count)
                t.update()

        # print('Epoch {} - loss: {}, acc: {}'.format(i, total_loss / count, total_acc / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_id', total_loss_id / count, i)
        writer.add_scalar('train/loss_quadrant', total_loss_quadrant / count, i)
        writer.add_scalar('train/acc', total_acc / count, i)
        writer.add_scalar('train/quad_acc', total_quad_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_acc = 0
            total_quad_acc = 0
            total_loss_id = 0
            total_loss_quadrant = 0
            count = 0
            with tqdm(total=len(test_loader)) as t:
                t.set_description('-- testing')
                for batch_i, batch_data in enumerate(test_loader):
                    patches, labels, masks, _ = batch_data

                    cls, losses, cls_acc, quad_acc = model_func(patches, labels, masks)

                    loss_cls = losses[0] + losses[1]

                    cls = cls.data.cpu().numpy()

                    # if batch_i < 3:
                    #     for patch_i in range(0, len(patches), 2):
                    #         nd_data = patches[patch_i].data.cpu().numpy()
                    #         nd_cls = cls[patch_i]
                    #         nd_cls = np.argmax(nd_cls, axis=0)
                    #         color = np.zeros((nd_data.shape[0], 5))
                    #         for pt in range(nd_data.shape[0]):
                    #             color[pt, 0:3] = colors.get_color(nd_cls, True) if nd_data[pt, 6] > 0.5 else np.ones(
                    #                 (3,)) * 0.5
                    #             color[pt, 3] = 1
                    #             color[pt, 4] = nd_cls
                    #
                    #         np.savetxt(
                    #             f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_cls/{batch_i}_{patch_i}.txt',
                    #             np.concatenate((nd_data[:, 0:6], color), axis=1))

                    total_loss += loss_cls.item()
                    total_acc += cls_acc.item()
                    total_quad_acc += quad_acc.item()
                    total_loss_id += losses[0].item()
                    total_loss_quadrant += losses[1].item()
                    count += 1
                    t.set_postfix(acc=cls_acc.item(), quad_acc=quad_acc.item(), total_acc=total_acc / count)
                    t.update()

            # print('  |-- test loss: {}, acc: {}'.format(total_loss / count, total_acc / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_id', total_loss_id / count, i)
            writer.add_scalar('test/loss_quadrant', total_loss_quadrant / count, i)
            writer.add_scalar('test/acc', total_acc / count, i)
            writer.add_scalar('test/quad_acc', total_quad_acc / count, i)

            if best_acc <= total_acc / count:
                os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(
                               experiment_dir, 'snapshots', 'model_best'))
                best_acc = total_acc / count

        if i == 100:
            lr = lr * 0.5
            optimizer = optim.Adam(
                model.parameters(), lr=lr
            )

        os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(experiment_dir, 'snapshots', 'model_latest'))

        if i % 20 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(experiment_dir, 'snapshots', 'model_{}'.format(i)))

        writer.flush()
