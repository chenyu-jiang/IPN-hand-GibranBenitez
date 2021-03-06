import argparse
from pathlib import Path

from . import spatial_transforms


class BoolFlagAction(argparse.Action):
    """Action that stores bool flag depending on whether --option or --no-option is passed"""

    def __init__(self,
                 option_strings,
                 dest,
                 default=False,
                 required=False,
                 help=None):
        option_strings = option_strings + [s.replace('--', '--no-') for s in option_strings]
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        val = not option_string.startswith('--no')
        setattr(namespace, self.dest, val)


def get_argument_parser():
    parser = argparse.ArgumentParser("Action recognition")

    add_common_args(parser)
    add_path_args(parser)
    add_model_args(parser)
    add_dataset_args(parser)
    add_input_args(parser)
    add_training_args(parser)

    return parser


def add_common_args(parser):
    parser.add_argument(
        '--train',
        action=BoolFlagAction,
        default=True,
        help='Whether training should be performed'
    )
    parser.add_argument(
        '--val',
        action=BoolFlagAction,
        default=True,
        help='Whether validation should be performed'
    )
    parser.add_argument(
        '--test',
        action=BoolFlagAction,
        default=True,
        help='Whether testing should be performed'
    )
    parser.add_argument(
        '--onnx', type=str, metavar='OUTPUT_ONNX_FILE',
        help='Export to ONNX model with a given name and exit'
    )
    parser.add_argument(
        '--sync-bn',
        action=BoolFlagAction,
        help='Replace all synchronized batchnorm operations with synchronized batchnorm'
    )
    parser.add_argument(
        '--fp16',
        action=BoolFlagAction,
        help='Enable training and validation in half precision'
    )
    parser.add_argument(
        '--tta',
        action=BoolFlagAction,
        help='Enable test time augmentations. Testing may take longer'
    )
    parser.add_argument(
        '--manual-seed',
        default=1,
        type=int,
        help='Manually set random seed'
    )
    parser.add_argument(
        '--cuda',
        action=BoolFlagAction,
        default=True,
        help='Whether cuda should be used'
    )
    parser.add_argument(
        '-j', '--n-threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading'
    )
    parser.add_argument(
        '--softmax-in-test',
        action=BoolFlagAction,
        help='Normalize outputs in testing before argmax'
    )

    parser.add_argument(
        '--checkpoint',
        default=5,
        type=int,
        help='Number of epochs to train before new checkpoint is saved'
    )
    parser.add_argument(
        '--resume-train',
        action=BoolFlagAction,
        default=True,
        help='Restore optimizer state and start epoch when loading checkpoint'
    )

    parser.add_argument(
        '--try-resume',
        action=BoolFlagAction,
        default=True,
        help='Attempts to find latest checkpoint in RESULT_PATH'
    )


def add_model_args(parser):
    group = parser.add_argument_group("Model")
    group.add_argument(
        '--model',
        default='resnet34_vtn',
        type=str,
        help='Either full model name in form <encoder>_<type> or model type (in this case encoder option should be '
             'provided). For example: resnet34_vtn, mobilenetv2_vtn, resnet34_vtn_two_stream'
    )
    group.add_argument(
        '--n-classes',
        type=int,
        help='Number of classes (for example: kinetics: 400, ucf101: 101, hmdb51: 51)'
    )
    group.add_argument(
        '--encoder',
        default='resnet34',
        type=str,
        help='Encoder used in model'
    )
    group.add_argument(
        '--model-depth',
        default=18,
        type=int,
        choices=[10, 18, 34, 50, 101],
        help='Depth of 3D-ResNet model'
    )
    group.add_argument(
        '--fusion-mode',
        default='cat1',
        type=str,
        choices=['sum', 'cat1', 'cat2'],
        help='Fusion mode for vtn two stream encoder'
    )
    group.add_argument(
        '--resnet-shortcut',
        default='B',
        choices=['A', 'B'],
        type=str,
        help='Shortcut type of resnet'
    )
    group.add_argument(
        '--wide-resnet-k',
        default=2,
        type=int,
        help='Wide resnet k'
    )
    group.add_argument(
        '--resnext-cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality'
    )
    group.add_argument(
        '--hidden-size',
        default=512,
        type=int,
        help='Size of hidden state and cell state of lstm'
    )
    group.add_argument(
        '--input-channels',
        default=1,
        type=int,
        help='Number of network input channels'
    )
    group.add_argument(
        '--bidirectional-lstm',
        action=BoolFlagAction,
        help='Make LSTM layers bidirectional'
    )
    group.add_argument(
        '--layer-norm',
        action=BoolFlagAction,
        help='Use LayerNormalization in VTN',
        default=True
    )
    group.add_argument(
        '--recurrent-vtn',
        action=BoolFlagAction,
        help='Use recurrent VTN decoder block i.s.o. stacked blocks',
        default=False
    )
    group.add_argument(
        '--num-stages',
        default=1,
        type=int,
        help='Number of network stages (used for MS-TCN)'
    )
    group.add_argument(
        '--num-layers',
        default=1,
        type=int,
        help='Number of network layers (used for MS-TCN, LSTM, GRU)'
    )
    group.add_argument(
        '--num-f-maps',
        default=64,
        type=int,
        help='Number of feature maps (used for MS-TCN)'
    )
    group.add_argument(
        '--causal-config',
        default="none",
        help='Select causal convolution config (used for MS-TCN)'
    )


def add_input_args(parser):
    group = parser.add_argument_group("Input")
    group.add_argument(
        '--sample-size',
        default=[32],
        nargs='+',
        type=int,
        help='Height (first) and Width (second) of inputs. If only one value is given, it is used for height and width'
    )
    group.add_argument(
        '--sample-rotation',
        default=0,
        type=int,
        help='Enables random rotation during training. Rotation range is (-sample-rotation,+sample-rotation)'
    )
    group.add_argument(
        '--sample-duration', '--seq', '--clip-size',
        default=48,
        type=int,
        help='Temporal duration of input clips'
    )
    group.add_argument(
        '--temporal_stride', '--st',
        default=1, type=int,
        help='Frame skip rate of sampled clips'
    )
    group.add_argument(
        '--temporal-shift',
        default=0.0, type=float,
        help='Shift clips randomly back and forth during training.'
    )
    group.add_argument(
        '--temporal-shift-mode',
        default='pad', type=str,
        help='Shift clips randomly back and forth during training.'
    )
    group.add_argument(
        '--temporal-scale',
        default=0.0, type=float,
        help='Scales clips randomly in the temporal domain. A value of 0.2 will scale in range (0.8, 1.2)'
    )
    group.add_argument(
        '--initial-scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping'
    )
    group.add_argument(
        '--n-scales',
        default=4,
        type=int,
        help='Number of scales for multiscale cropping'
    )
    group.add_argument(
        '--scale-step',
        default=2 ** (-1 / 4),
        type=float,
        help='Scale step for multiscale cropping'
    )
    group.add_argument(
        '--mean-dataset',
        default='imagenet',
        choices=list(spatial_transforms.MEAN_STATISTICS.keys()),
        type=str,
        help='Which dataset to use for mean subtraction'
    )
    group.add_argument(
        '--mean-norm',
        default=True,
        action=BoolFlagAction,
        help='Should inputs be normalizaed by mean'
    )
    group.add_argument(
        '--std-norm',
        default=True,
        action=BoolFlagAction,
        help='Should inputs be normalized by std'
    )
    group.add_argument(
        '--hflip',
        default=False,
        action=BoolFlagAction,
        help='Should horizontal flipping be performed for autgumentation'
    )
    group.add_argument(
        '--drop-last',
        default=True,
        action=BoolFlagAction,
        help='Drop the last batch if it is incomplete.'
    )
    group.add_argument(
        '--max-pixel-value',
        default=255,
        type=int,
        help='Input pixel range maximum (determines max value before conversion to tensor)'
    )
    group.add_argument(
        '--min-pixel-value',
        default=0,
        type=int,
        help='Input pixel range minimum (determines min value before conversion to tensor)'
    )
    group.add_argument(
        '--contrast',
        default=0.0,
        type=float,
        help='Set random contrast scaling during training. E.g: 0.2 will scale the values with a factor between 0.8 and'
             '1.2'
    )
    group.add_argument(
        '--brightness',
        default=0.0,
        type=float,
        help='Set random brightness offset during training. E.g: 0.1 will randomly add/subtract 10%% of the value range'
    )
    group.add_argument(
        '--gauss-noise',
        default=0.0,
        type=float,
        help='Enable photometric distortion by adding gaussian noise'
    )
    group.add_argument(
        '--scale-in-test',
        default=1.0,
        type=float,
        help='Spatial scale in test'
    )
    group.add_argument(
        '--crop-position-in-test',
        default='c',
        choices=['c', 'tl', 'tr', 'bl', 'br'],
        type=str,
        help='Cropping position in test'
    )


def add_path_args(parser):
    group = parser.add_argument_group("Path")
    group.add_argument(
        '--root-path',
        type=Path,
        help='Root directory path of data. Will be used as base directory for video path, annotation path, etc.'
    )
    group.add_argument(
        '--video-path',
        type=Path,
        help='Path to directory with video files'
    )
    group.add_argument(
        '--annotation-path',
        type=Path,
        help='Annotation file path'
    )
    group.add_argument(
        '--result-path',
        metavar='RESULT_PATH',
        default='results',
        type=Path,
        help='Where to store training results (logs, checkpoints, etc.)'
    )
    group.add_argument(
        '--rgb-path',
        type=Path,
        help='Path to RGB model weights (fo two stream models)'
    )
    group.add_argument(
        '--motion-path',
        type=Path,
        help='Path to motion model weights (for two stream models)'
    )
    group.add_argument(
        '--resume-path',
        type=Path,
        help='Path to checkpoint to resume training'
    )
    group.add_argument(
        '--pretrain-path',
        type=Path,
        help='Path to pretrained model (.pth)'
    )


def add_dataset_args(parser):
    group = parser.add_argument_group("Data")
    group.add_argument(
        '--dataset',
        default='kinetics',
        type=str,
        help='Dataset name, possibly with split number (e.g. kinetics, ucf101_1)'
    )
    group.add_argument(
        '--dataset-config',
        type=str,
        help='Path to dataset configuration file'
    )
    group.add_argument(
        '--split',
        default=1,
        type=int,
        help='Dataset split number'
    )
    group.add_argument(
        '--test-subset',
        default='val',
        choices=('val', 'test'),
        help='Used subset in test (val | test)'
    )
    group.add_argument(
        '--n-val-clips',
        default=3,
        type=int,
        help='Number of clips to be sampled from video in validation'
    )
    group.add_argument(
        '--n-test-clips',
        default=10,
        type=int,
        help='Number of clips to be sampled from video in testing'
    )
    group.add_argument(
        '--video-format',
        default='frames',
        choices=['video', 'frames'],
        help='In what format dataset is stored'
    )
    group.add_argument(
        '--weighted-sampling',
        action=BoolFlagAction,
        help='when this option is set, clips will be sampled with probability proportional to its label '
             '"class_weights" from dataset config if it is provided or with class balanced probability otherwise'
    )


def add_training_args(parser):
    group = parser.add_argument_group("Training")
    group.add_argument(
        '-b', '--batch-size', '--batch',
        default=8,
        type=int,
        metavar='BATCH_SIZE',
        help='Batch Size'
    )
    group.add_argument(
        '--iter-size',
        default=1,
        type=int,
        metavar='ITER_SIZE',
        help='How many batches will be forwarded before parameter update (i.e. effective batch size will be '
             'BATCH_SIZE * ITER_SIZE'
    )
    group.add_argument(
        '--optimizer', '--solver',
        default='adam',
        type=str.lower,
        choices=['sgd', 'adam'],
        help='Which optimizer to use.'
    )
    group.add_argument(
        '--learning-rate', '--lr',
        default=0.1,
        type=float,
        help='Initial learning rate'
    )
    group.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum'
    )
    group.add_argument(
        '--dampening',
        default=0.9,
        type=float,
        help='dampening of SGD'
    )
    group.add_argument(
        '--weight-decay',
        default=1e-4,
        type=float,
        help='Weight Decay'
    )
    group.add_argument(
        '--nesterov',
        default=True,
        action=BoolFlagAction,
        help='Use Nesterov momentum'
    )
    group.add_argument(
        '--scheduler', '--lr-policy',
        default='plateau',
        choices=['step', 'plateau'],
        help='Learning rate decay policy.'
    )
    group.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='Factor by which learning rate will be decayed'
    )
    group.add_argument(
        '--lr-patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    group.add_argument(
        '--lr-step-size',
        default=10,
        type=int,
        help='Period of learning rate decay'
    )
    group.add_argument(
        '--n-epochs',
        default=150,
        type=int,
        help='Total number of epochs to train'
    )
    group.add_argument(
        '--begin-epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    group.add_argument(
        '--teacher-model',
        type=str,
        help='Model (in the same format as in --model) that will be used as teacher in knowledge distillation (if '
             'option is passed)'
    )
    group.add_argument(
        '--teacher-checkpoint',
        type=Path,
        help='Path to teacher model checkpoint'
    )
    group.add_argument(
        '--gradient-clipping',
        type=float,
        help='Gradients will be clipped to this value (if passed)'
    )
    group.add_argument(
        '--ctc-loss',
        action=BoolFlagAction,
        default=False,
        help='Enable CTC loss instead of cross entropy loss'
    )
    group.add_argument(
        '--ctc-loss-type',
        type=str,
        default='ctc',
        choices=['ctc', 'ctfl_cs', 'ctfl_sp'],
        help='CTC loss function flavor'
    )
    group.add_argument(
        '--ctc-gamma',
        type=float,
        default=1.0,
        help='Focus parameter for CTFL_cs and CTFL_sp loss functions'
    )


def parse_arguments():
    parser = get_argument_parser()
    args = parser.parse_args()

    # make sure sample_size is a tulpe of 2 elements
    if len(args.sample_size) == 1:
        args.sample_size = args.sample_size * 2
    args.sample_size = tuple(args.sample_size)

    return args
