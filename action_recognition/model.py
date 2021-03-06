import torch
from torch import nn

from .models import (densenet_3d, inception_i3d, lstm_attention, ms_tcn,
                     multi_frame_baseline, video_transformer, motion, two_stream)
from .models.modules.sync_batchnorm import (DataParallelWithCallback,
                                            SynchronizedBatchNorm2d)
from .models.r3d import R3D_MODELS
from .utils import load_state

MODEL_REGISTRY = {
    'vtn': lambda args, encoder: video_transformer.VideoTransformer(
        args.hidden_size,
        args.sample_duration,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        num_layers=args.num_layers,
        layer_norm=args.layer_norm,
        recurrent=args.recurrent_vtn,
        input_channels=args.input_channels,
    ),
    'ms_tcn': lambda args, encoder: ms_tcn.MultiStageTemporalConvNet(
        args.hidden_size,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        input_channels=args.input_channels,
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        num_f_maps=args.num_f_maps,
        causal_config=args.causal_config
    ),
    'ms_tcn_two_stream': lambda args, encoder: two_stream.TwoStreamNetwork(
        ms_tcn.MultiStageTemporalConvNet(
            args.hidden_size,
            encoder,
            args.n_classes,
            args.sample_size,
            False if args.pretrain_path or args.resume_path else True,
            input_channels=args.input_channels,
            num_stages=args.num_stages,
            num_layers=args.num_layers,
            num_f_maps=args.num_f_maps,
            causal_config=args.causal_config
        ),
        motion_mode='rgbdiff',
        path=args.rgb_path,
        motion_path=args.motion_path
    ),
    'lstm': lambda args, encoder: lstm_attention.VisualAttentionLSTM(
        args.hidden_size,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        input_channels=args.input_channels,
        use_attention=False,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional_lstm,
    ),
    'lstm_two_stream': lambda args, encoder: two_stream.TwoStreamNetwork(
        lstm_attention.VisualAttentionLSTM(
            args.hidden_size,
            encoder,
            args.n_classes,
            args.sample_size,
            False if args.pretrain_path or args.resume_path else True,
            input_channels=args.input_channels,
            use_attention=False,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional_lstm,
        ),
        motion_mode='rgbdiff',
        path=args.rgb_path,
        motion_path=args.motion_path
    ),
    'gru': lambda args, encoder: lstm_attention.VisualAttentionLSTM(
        args.hidden_size,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        input_channels=args.input_channels,
        use_attention=False,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional_lstm,
        use_gru=True,
    ),
    'attn_lstm': lambda args, encoder: lstm_attention.VisualAttentionLSTM(
        args.hidden_size,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        input_channels=args.input_channels,
        use_attention=True,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional_lstm,
    ),
    'attn_gru': lambda args, encoder: lstm_attention.VisualAttentionLSTM(
        args.hidden_size,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        input_channels=args.input_channels,
        use_attention=True,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional_lstm,
        use_gru=True,
    ),
    'vtn_encoder': lambda args, encoder: video_transformer.VideoTransformerEncoder(
        args.hidden_size,
        args.sample_duration,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        layer_norm=args.layer_norm,
        input_channels=args.input_channels,
    ),
    'vtn_decoder': lambda args, encoder: video_transformer.VideoTransformerDecoder(
        args.hidden_size,
        args.sample_duration,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        layer_norm=args.layer_norm,
    ),
    'vtn_two_stream': lambda args, encoder: two_stream.TwoStreamNetwork(
        video_transformer.VideoTransformer(
            args.hidden_size,
            args.sample_duration-1,
            encoder,
            args.n_classes,
            args.sample_size,
            False if args.pretrain_path or args.resume_path else True,
            input_channels=args.input_channels,
            layer_norm=args.layer_norm,
            recurrent=args.recurrent_vtn,
        ),
        motion_mode='rgbdiff',
        path=args.rgb_path,
        motion_path=args.motion_path
    ),
    'baseline': lambda args, encoder: multi_frame_baseline.MultiFrameBaseline(
        args.sample_duration,
        encoder,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        input_channels=args.input_channels,
    ),
    'resnet34_attn_single': lambda args, encoder: lstm_attention.ResnetAttSingleInput(
        args.hidden_size,
        args.sample_duration,
        args.n_classes,
        args.sample_size,
        False if args.pretrain_path or args.resume_path else True,
        resnet_size=34
    ),
    'inception_i3d': lambda args, encoder: inception_i3d.InceptionI3D(
        num_classes=args.n_classes
    ),
    'densenet201': lambda args, encoder: densenet_3d.densenet201(
        sample_size=args.sample_size,
        sample_duration=args.sample_duration,
        num_classes=400
    )
    ,
    **R3D_MODELS,
}


def make_bn_synchronized(bn_module):
    """Convert all BatchNorm modules to SynchronizedBatchNorm"""
    new_module = SynchronizedBatchNorm2d(bn_module.num_features, eps=bn_module.eps, momentum=bn_module.momentum,
                                         affine=bn_module.affine)

    if new_module.track_running_stats:
        new_module.running_mean = bn_module.running_mean
        new_module.running_var = bn_module.running_var
        new_module.num_batches_tracked = bn_module.num_batches_tracked
    new_module.weight = bn_module.weight
    new_module.bias = bn_module.bias
    return new_module


def _replace_bns(model: nn.Module, memo=None):
    if memo is None:
        memo = set()
    if model not in memo:
        memo.add(model)
        for name, module in model._modules.items():
            if module is None:
                continue
            if isinstance(module, nn.BatchNorm2d):
                if isinstance(model, nn.Sequential):
                    model._modules[name] = make_bn_synchronized(module)
                else:
                    setattr(model, name, make_bn_synchronized(module))
            _replace_bns(module, memo)
    return model


def create_model(args, model, pretrain_path=None):
    """Construct model with a given name and args.

    Args:
        args (Namespace): Options for model construction
        model (str): Name of the model in ENCODER_DECODER or DECODER format.
        pretrain_path (Path): Path to a checkpoint with the pretrained model.
    """
    model = model.replace("self_attn", "vtn")
    if len(model.split('_')) > 1:
        encoder_name, model_type = model.split('_', 1)
    else:
        encoder_name = args.encoder
        model_type = model
    encoder_name = encoder_name.replace('-', '_')

    if model in MODEL_REGISTRY:
        # if model with exactly same name is known
        model = MODEL_REGISTRY[model](args, encoder_name)
    else:
        model = MODEL_REGISTRY[model_type](args, encoder_name)

    # load pre-trained model
    if pretrain_path:
        print('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(str(args.pretrain_path))

        if hasattr(model, 'load_checkpoint'):
            model.load_checkpoint(pretrain['state_dict'])
        else:
            load_state(model, pretrain['state_dict'])

    if args.cuda:
        model = model.cuda()
        if args.sync_bn:
            model = _replace_bns(model)
            wrapped_model = DataParallelWithCallback(model)
        else:
            wrapped_model = nn.DataParallel(model)
    else:
        wrapped_model = model

    if args.fp16:
        model = model.half()

        # do not train batchnorms in FP16 precision
        def _float_bns(layer):
            if isinstance(layer, (nn.BatchNorm2d,)):
                layer.float()

        model.apply(_float_bns)

    parameters = model.trainable_parameters()
    return wrapped_model, parameters
