import argparse
import os.path
import sys
import shutil
import signal
import glob
import torch
from loguru import logger
import pickle

from torch.utils.data import DataLoader
import torch.optim as optim
from src.data.TeethTrainingDataset import TeethTrainingDataset
import src.utils.run_train as run_train
from src.utils.tensorboard_utils import TensorboardUtils
from src.models import *

# =============== Global variables ===============
torch.random.manual_seed(20000228)
torch.cuda.manual_seed(20000228)
exp_out_dir = ''

initial_checkpoint_data = {
    'stage_1': {
        'epochs': 0,
        'lr': 0.001,
        'lr_decay': 0.5,
        'weight_decay': 0,
        'decay_step': 100,
    },
    'stage_2': {
        'epochs': 0,
        'lr': 0.001,
        'lr_decay': 0.1,
        'weight_decay': 0,
        'decay_step': 500,
    }
}

checkpoint_data = initial_checkpoint_data

# Record if the checkpoint is updated, in case of repeated saving
# Update it manually
checkpoint_data_updated = {
    'stage_1': False,
    'stage_2': False
}

models = [
    [CentroidPredictionNet()],
    [AllToothSegNet(4), AllToothSegNet(5)]
]

model_name_of_stage = [
    ['model_pred_centroid'],
    ['model_tooth_seg', 'model_tooth_seg_refine']
]


# ============= Program init methods =============
def init_logger():
    logger.remove()
    logger.add(sys.stdout, colorize=True, enqueue=True, backtrace=True, diagnose=True)


def make_arg_parser() -> argparse.ArgumentParser:
    """
    Prepare CLI arg parser

    :return: argparse instance
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # Basic information
    parser.add_argument('--data_dir', '-d', type=str, help='Path of dataset directory', required=False,
                        default='../data')
    parser.add_argument('--out_dir', '-o', type=str, help='Path of output directory', required=False,
                        default='../output')
    parser.add_argument('--name', '-n', type=str, help='Name of this experiment', required=False,
                        default='experiment_0')
    parser.add_argument('--backup', '-b', help='Make a backup of current state to output directory',
                        required=False, default=False, action='store_true')
    parser.add_argument('--stages', type=int, help='Which stages to train',
                        required=False, default=[1, 2], nargs='+')
    # Training hyperparams
    parser.add_argument('--batch_size', type=int, help='Size of a batch', required=False, default=32)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs', required=False,
                        default=[3000, 3000], nargs='+')
    return parser


def check_and_prepare_files() -> None:
    """
    Check and prepare needed files
    """
    global exp_out_dir
    global checkpoint_data

    exp_out_dir = os.path.abspath(os.path.join(ARGS.out_dir, ARGS.name))
    logger.info('Experiment output directory: {}'.format(exp_out_dir))
    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)
    else:
        load_checkpoint()
        logger.info('Find existing checkpoint: {}', checkpoint_data)

    if ARGS.backup:
        backup_dir = os.path.join(exp_out_dir, 'codes')
        shutil.copytree('.', backup_dir, dirs_exist_ok=True)


def create_summary_writer():
    # Call constructor only is enough
    _ = TensorboardUtils(exp_out_dir)


# =========== Data persistence methods ===========
def save_checkpoint() -> None:
    global exp_out_dir
    global checkpoint_data

    checkpoint_path = os.path.join(exp_out_dir, 'snapshots', 'checkpoint.pkl')
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path))

    logger.info('Saving checkpoint to [{}]', checkpoint_path)

    file = open(checkpoint_path, 'wb')
    pickle.dump(checkpoint_data, file)
    file.close()

    # Save model state dicts
    for stage in range(2):
        if checkpoint_data_updated['stage_' + str(stage + 1)] \
                and checkpoint_data['stage_' + str(stage + 1)]['epochs'] > 0:
            for i in range(len(model_name_of_stage[stage])):
                filename = '{}_{}'\
                    .format(model_name_of_stage[stage][i], checkpoint_data['stage_' + str(stage + 1)]['epochs'])
                torch.save(models[stage][i].state_dict(), os.path.join(exp_out_dir, 'snapshots', filename))


def find_most_possible_model_path(checkpoint, model_name):
    """
    Find the most possible model path to load

    Firstly check checkpoint_data.epochs and checkpoint_data.epochs - 1, to find the
    latest saved model. If not found, then select according to the maximum epoch-suffix.
    If no model available, return None.

    :param model_name: The name of the model
    :type model_name: str
    :return: Accurate model path and the corresponding epochs
    :rtype: str | None, int
    """
    basename = os.path.join(exp_out_dir, 'snapshots', model_name)
    possible_names = glob.glob(basename + '_' + str(checkpoint['epochs']))
    if len(possible_names) > 0:
        return possible_names[0], checkpoint['epochs']

    possible_names = glob.glob(basename + '_' + str(checkpoint['epochs'] - 1))
    if len(possible_names) > 0:
        return possible_names[0], checkpoint['epochs'] - 1

    possible_names = glob.glob(basename + '_*')
    if len(possible_names) > 0:
        latest_name, latest_epochs = possible_names[0]
        for name in possible_names:
            epochs = int(name.replace(basename + '_', ''))
            if epochs > latest_epochs:
                latest_epochs = epochs
                latest_name = name
        return latest_name, latest_epochs
    return None, 0


def load_checkpoint() -> None:
    global checkpoint_data
    if os.path.exists(os.path.join(exp_out_dir, 'snapshots', 'checkpoint.pkl')):
        file = open(os.path.join(exp_out_dir, 'snapshots', 'checkpoint.pkl'), 'rb')
        checkpoint_data = dict(**pickle.load(file))
        file.close()

    for stage in range(2):
        stage_name = 'stage_' + str(stage + 1)

        for i in range(len(model_name_of_stage[stage])):
            model_path, epochs = \
                find_most_possible_model_path(checkpoint_data[stage_name], model_name_of_stage[stage][i])
            logger.info('Find model for {}: {}, epochs: {}'.format(model_name_of_stage[stage][i], model_path, epochs))
            if model_path is None:
                logger.warning('No model found for {}, will train from scratch'.format(model_name_of_stage[stage][i]))
                checkpoint_data[stage_name] = initial_checkpoint_data[stage_name]
            else:
                models[stage][i].load_state_dict(torch.load(model_path))
                checkpoint_data[stage_name]['epochs'] = epochs
                models[stage][i].cuda()


# ========== Exception handling methods ==========
def on_abnormal_exit(sig, frame) -> None:
    """
    Triggered when the program exits abnormally.

    This usually happens when the process is sent a SIGINT/SIGTERM/SIGABRT etc.
    Under that circumstance, training procedure must be saved immediately, in case of
    loss of training process.

    :param sig: Signal received
    :type sig: int
    :param frame: Stack frame
    :type frame: frame
    """
    logger.warning('Receive signal: {}', sig)
    save_checkpoint()
    exit(0)


def register_signal_handler() -> None:
    signal.signal(signal.SIGINT, on_abnormal_exit)
    signal.signal(signal.SIGTSTP, on_abnormal_exit)
    signal.signal(signal.SIGQUIT, on_abnormal_exit)


if __name__ == '__main__':
    global model

    ##########################################
    # Init
    ##########################################
    init_logger()
    register_signal_handler()

    parser = make_arg_parser()
    ARGS = parser.parse_args()
    logger.info(ARGS)
    check_and_prepare_files()
    create_summary_writer()

    ##########################################
    # Data loader
    ##########################################
    train_set = TeethTrainingDataset(os.path.join(ARGS.data_dir, 'train.list'), train=True, remove_curvature=True)
    test_set = TeethTrainingDataset(os.path.join(ARGS.data_dir, 'test.list'), train=False, remove_curvature=True)
    train_loader = DataLoader(
        train_set,
        batch_size=ARGS.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        test_set,
        batch_size=ARGS.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    logger.info('Dataset load complete')

    ##########################################
    # Model
    ##########################################

    # Stage 1: centroid prediction
    optimizer = optim.Adam(
        models[0][0].parameters(), lr=checkpoint_data['stage_1']['lr'],
        weight_decay=checkpoint_data['stage_1']['weight_decay'],
        amsgrad=True
    )

    ##########################################
    # Train
    ##########################################
    models[0][0].cuda()
    # models[1][0].cuda()
    # models[1][1].cuda()

    best_loss = 10000
    if ARGS.stages.__contains__(1):
        logger.info('................ Training stage 1 ................')
        n_epochs = ARGS.n_epochs[0] if type(ARGS.n_epochs) == list and len(ARGS.n_epochs) > 0 else ARGS.n_epochs
        checkpoint = checkpoint_data.get('stage_1')
        for epoch in range(checkpoint['epochs'], n_epochs):
            optimizer, new_lr, mean_loss_test = run_train.train_one_epoch_stage1(
                epoch, models[0][0], [train_loader, test_loader], optimizer,
                checkpoint['weight_decay'], checkpoint['lr'], checkpoint['lr_decay'], checkpoint['decay_step']
            )
            checkpoint_data['stage_1']['epochs'] = epoch
            checkpoint_data['stage_1']['lr'] = new_lr
            checkpoint_data_updated['stage_1'] = True

            if best_loss >= mean_loss_test:
                best_loss = mean_loss_test
                should_save = True
            else:
                should_save = (epoch % 100 == 0)

            if should_save:
                save_checkpoint()

        # Save checkpoint when training stopped
        save_checkpoint()
        # Training done, do not save stage_1 anymore
        checkpoint_data_updated['stage_1'] = False

    if ARGS.stages.__contains__(2):
        logger.info('................ Training stage 2 ................')
        n_epochs = ARGS.n_epochs[-1] if type(ARGS.n_epochs) == list and len(ARGS.n_epochs) > 0 else ARGS.n_epochs
        checkpoint = checkpoint_data.get('stage_2')
        for epoch in range(checkpoint['epochs'], n_epochs):
            optimizer, new_lr, should_save = run_train.train_one_epoch_stage2(
                epoch, models[1], models[0][0], [train_loader, test_loader], optimizer,
                checkpoint['weight_decay'], checkpoint['lr'], checkpoint['lr_decay'], checkpoint['decay_step'])
            checkpoint_data['stage_2']['epochs'] = epoch
            checkpoint_data['stage_2']['lr'] = new_lr

            checkpoint_data_updated['stage_1'] = ARGS.stages.__contains__(1)
            checkpoint_data_updated['stage_2'] = True

            if should_save:
                save_checkpoint()

        # Save checkpoint when training stopped
        save_checkpoint()
        # Training done, do not save stage_2 anymore
        checkpoint_data_updated['stage_2'] = False
