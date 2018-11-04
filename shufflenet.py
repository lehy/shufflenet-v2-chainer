import chainer
import chainer.links as L
import chainer.functions as F
import functools as fun
import logging
log = logging.getLogger(__name__)


def seq(name, *layers):
    """Create a Sequential with a name. This is done by creating a class
       deriving chainer.Sequential on the fly.

       (Not sure this is worth the obfuscation cost. This is just a
       bit nicer when you print() the sequentials.)

    """

    def __init__(self, *layers):
        chainer.Sequential.__init__(self, *layers)

    cls = type(name, (chainer.Sequential, ), dict(__init__=__init__))
    return cls(*layers)


def _1x1_conv_bn_relu(out_channels):
    """1x1 Conv + BN + ReLU.
    """
    return seq(
        "_1x1_conv_bn_relu",
        L.Convolution2D(
            None, out_channels, ksize=1, stride=1, pad=0, nobias=True),
        L.BatchNormalization(axis=(0, 2, 3)), F.relu)


def _3x3_dwconv_bn(stride):
    """3x3 DWConv (depthwise convolution) + BN.
    """
    return seq(
        "_3x3_dwconv_bn",
        L.DepthwiseConvolution2D(
            None,
            channel_multiplier=1,
            ksize=3,
            stride=stride,
            pad=1,
            nobias=True), L.BatchNormalization(axis=(0, 2, 3)))


class ShuffleNetV2BasicBlock(chainer.Chain):
    """ShuffleNet v2 basic unit. Split and concat. Fig 3 (c), page 7.

    """

    def __init__(self, out_channels):
        super().__init__()
        with self.init_scope():
            branch_channels = out_channels // 2
            assert 2 * branch_channels == out_channels, (
                "ShuffleNetV2BasicBlock out_channels must be divisible by 2")
            # Note that in the basic block the 3x3 dwconv has stride=1.
            self.branch = chainer.Sequential(
                _1x1_conv_bn_relu(branch_channels), _3x3_dwconv_bn(stride=1),
                _1x1_conv_bn_relu(branch_channels))

    def forward(self, x):
        x1, x2 = F.split_axis(x, 2, axis=1)
        x2 = self.branch(x2)
        return channel_shuffle(F.concat((x1, x2), axis=1))


def channel_shuffle(x, groups=2):
    n, c, h, w = x.shape
    x = x.reshape(n, groups, c // groups, h, w)
    x = x.transpose(0, 2, 1, 3, 4)
    x = x.reshape(n, c, h, w)
    return x


class ShuffleNetV2DownsampleBlock(chainer.Chain):
    """ShuffleNet v2 unit for spatial downsampling. Fig 3 (d), page 7.

    """

    def __init__(self, out_channels):
        super().__init__()
        with self.init_scope():
            branch_channels = out_channels // 2
            assert 2 * branch_channels == out_channels, (
                "ShuffleNetV2BasicBlock out_channels must be divisible by 2")
            # Note that in the downsampling block the 3x3 dwconv has stride=2.
            self.branch1 = chainer.Sequential(
                _3x3_dwconv_bn(stride=2), _1x1_conv_bn_relu(branch_channels))
            self.branch2 = chainer.Sequential(
                _1x1_conv_bn_relu(branch_channels), _3x3_dwconv_bn(stride=2),
                _1x1_conv_bn_relu(branch_channels))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return channel_shuffle(F.concat((x1, x2), axis=1))


def ShuffleNetV2Stage(out_channels, repeat):
    assert repeat > 1, "ShuffleNetV2Stage: repeat must be > 1"
    first = ShuffleNetV2DownsampleBlock(out_channels)
    rest = ShuffleNetV2BasicBlock(out_channels)

    stage = seq("Stage")
    stage.append(first)
    stage += rest.repeat(repeat - 1)

    return stage


def log_size(x, name):
    """Insert this in a Sequential to get a debug output of the input size
    at a point in the sequence.

    """
    log.info("%s: x has shape %s", name, x.data.shape)
    return x


def ShuffleNetV2Features(k):
    """Create a ShuffleNet v2 network that computes a (1024, fw, fh)
       feature tensor.

       Parameter k is the multiplier giving the size of the
       network. Supported values are 0.5, 1, 1.5 and 2.

    """
    stage_repeats = [4, 8, 4]
    known_channels = {
        0.5: [24, 48, 96, 192, 1024],
        1.: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2: [24, 244, 488, 976, 1024],
    }
    try:
        channels = known_channels[k]
    except KeyError:
        raise KeyError("unsupported k={}, supported values are {}".format(
            k, known_channels.keys()))

    net = seq("Features")

    # Conv1
    conv1 = seq(
        "Conv1",
        L.Convolution2D(3, channels[0], ksize=3, stride=2, pad=1, nobias=True),
        L.BatchNormalization(axis=(0, 2, 3)), F.relu)
    net.append(conv1)

    max_pool = fun.partial(
        F.max_pooling_2d, ksize=3, stride=2, pad=1, cover_all=False)
    net.append(max_pool)

    # Stage2, Stage3, Stage4.
    for i_stage, stage_repeat in enumerate(stage_repeats):
        out_channels = channels[1 + i_stage]
        stage = ShuffleNetV2Stage(out_channels, stage_repeat)
        net.append(stage)

    # Conv5
    net.append(_1x1_conv_bn_relu(channels[-1]))

    return net


def ShuffleNetV2(k, out_size):
    """Create a ShuffleNet v2 network that outputs a vector of size
    out_size. It consists in global average pooling + fully connected
    layer on top of the feature network above.

    """
    # We do not flatten to ease reuse of features.
    return seq(
        "ShuffleNetV2",
        ShuffleNetV2Features(k),
        seq(
            "Head",
            # Global pooling (compute mean of each channel). We leave
            # keepdims=False to get a NxC array (with H and W removed and not
            # set to 1).
            fun.partial(F.mean, axis=(2, 3)),
            L.Linear(None, out_size)))


def test_shufflenet_features():
    import numpy as np
    with chainer.using_config('train', False):
        for k in [0.5, 1, 1.5, 2]:
            net = ShuffleNetV2Features(k)
            data = np.random.random((10, 3, 123, 567)).astype(np.float32)
            ret = net(data)
            assert ret.data.shape[:2] == (10, 1024)

            data = np.random.random((10, 3, 224, 224)).astype(np.float32)
            ret = net(data)
            assert ret.data.shape == (10, 1024, 7, 7)


def test_shufflenet():
    import numpy as np
    with chainer.using_config('train', False):
        for k in [0.5, 1, 1.5, 2]:
            for out_size in [2, 10, 1000]:
                net = ShuffleNetV2(k, out_size)
                data = np.random.random((10, 3, 123, 567)).astype(np.float32)
                ret = net(data)
                assert ret.data.shape == (10, out_size)


def test_use_features():
    import numpy as np
    with chainer.using_config('train', False):
        net = ShuffleNetV2(1, 2)
        features = net[0]
        data = np.random.random((10, 3, 123, 567)).astype(np.float32)
        ret = features(data)
        assert ret.data.shape[:2] == (10, 1024)
        data = np.random.random((10, 3, 224, 224)).astype(np.float32)
        ret = features(data)
        assert ret.data.shape == (10, 1024, 7, 7)
