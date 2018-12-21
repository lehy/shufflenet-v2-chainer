import chainer
import chainer.links as L
import chainer.functions as F
import functools as fun
import collections
import logging
log = logging.getLogger(__name__)


def describe_variable(name, v):
    try:
        shape = v.shape
    except AttributeError:
        shape = "(?)"  # "(shape not yet initialized)"
    return "{}{}".format(name, shape)


def describe_element(elt, prefix):
    if hasattr(elt, 'namedparams'):
        params = [
            describe_variable(name, variable)
            for name, variable in elt.namedparams()
        ]
        if len(params) > 2:
            prefix = prefix + '  '
            str_params = "\n{}".format(prefix) + ',\n{}'.format(prefix).join(
                params)
        else:
            str_params = ', '.join(params)
        return "{}({})".format(elt.__class__.__name__, str_params)
    else:
        return str(elt)


class Seq(chainer.Chain):
    def __init__(self, **named_layers):
        chainer.Chain.__init__(self)
        self._layers = collections.OrderedDict()
        for k, v in named_layers.items():
            self.add(k, v)

    def add(self, name, layer):
        with self.init_scope():
            if hasattr(self, name):
                raise KeyError("sequence {} already has attribute {}".format(
                    self, name))
            setattr(self, name, layer)
            self._layers[name] = layer

    def forward(self, x):
        for layer in self._layers.values():
            x = layer(x)
        return x

    def to_string(self, prefix=''):
        buf = ''
        for k, v in self._layers.items():
            if isinstance(v, Seq):
                buf += "{}{}\n{}".format(prefix, k,
                                         v.to_string(prefix=(prefix + '  ')))
            else:
                buf += "{}{}\t{}\n".format(prefix, k,
                                           describe_element(v, prefix))
        return buf

    def __str__(self):
        return self.to_string()


class BatchNormalization(L.BatchNormalization):
    def __init__(self, *args, **kwargs):
        L.BatchNormalization.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        # log.debug("batch norm finetune=%s", finetune)
        if 'finetune' not in kwargs:
            finetune = getattr(chainer.config, 'finetune', False)
            kwargs['finetune'] = finetune
        return L.BatchNormalization.__call__(self, *args, **kwargs)

    # problem:
    # - avg_mean and avg_var are initialized at None
    # - when not giving the size to BatchNotmalization, they get
    #   initialized at the beginning of forward()
    # - but they are always initialized on the cpu, even if one has called
    #   to_gpu() on the BN first
    # - so we get an error when running forward if the BN has been moved
    #   to gpu, since avg_mean and avg_var are still on cpu
    def _initialize_params(self, shape):
        dtype = self._dtype
        self.avg_mean = self._init_array(self._initial_avg_mean, 0, shape,
                                         dtype)  # Ronan
        self._initial_avg_mean = None
        self.register_persistent('avg_mean')
        self.avg_var = self._init_array(self._initial_avg_var, 1, shape,
                                        dtype)  # Ronan
        self._initial_avg_var = None
        self.register_persistent('avg_var')
        if self.gamma is not None:
            self.gamma.initialize(shape)
        if self.beta is not None:
            self.beta.initialize(shape)

    def _init_array(self, initializer, default_value, size, dtype):
        if initializer is None:
            initializer = default_value
        initializer = chainer.initializers._get_initializer(initializer)
        return chainer.initializers.generate_array(
            initializer, size, self.xp, dtype=dtype)


def _1x1_conv_bn_relu(out_channels):
    """1x1 Conv + BN + ReLU.
    """
    return Seq(
        conv=L.Convolution2D(
            None, out_channels, ksize=1, stride=1, pad=0, nobias=True),
        bn=BatchNormalization(axis=(0, 2, 3)),
        relu=F.relu)


def _3x3_dwconv_bn(stride):
    """3x3 DWConv (depthwise convolution) + BN.
    """
    return Seq(
        dwconv=L.DepthwiseConvolution2D(
            None,
            channel_multiplier=1,
            ksize=3,
            stride=stride,
            pad=1,
            nobias=True),
        bn=BatchNormalization(axis=(0, 2, 3)))


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
            self.branch = Seq(
                conv_bn_relu1=_1x1_conv_bn_relu(branch_channels),
                dwconv_bn=_3x3_dwconv_bn(stride=1),
                conv_bn_relu2=_1x1_conv_bn_relu(branch_channels))

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
            self.branch1 = Seq(
                dwconv_bn=_3x3_dwconv_bn(stride=2),
                conv_bn_relu=_1x1_conv_bn_relu(branch_channels))
            self.branch2 = Seq(
                conv_bn_relu1=_1x1_conv_bn_relu(branch_channels),
                dwconv_bn=_3x3_dwconv_bn(stride=2),
                conv_bn_relu2=_1x1_conv_bn_relu(branch_channels))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return channel_shuffle(F.concat((x1, x2), axis=1))


def ShuffleNetV2Stage(out_channels, repeat):
    assert repeat > 1, "ShuffleNetV2Stage: repeat must be > 1"

    stage = Seq()
    stage.add('downsample', ShuffleNetV2DownsampleBlock(out_channels))
    for i in range(repeat - 1):
        stage.add('basic{}'.format(i + 1),
                  ShuffleNetV2BasicBlock(out_channels))

    return stage


def log_size(x, name):
    """Insert this in a Sequential to get a debug output of the input size
    at a point in the sequence.

    """
    log.info("%s: x has shape %s", name, x.data.shape)
    return x


def run_dummy(model):
    """Run a dummy batch through a model.

    This is useful to initialize all arrays inside the model.  When
    sizes are not specified, some arrays are initialized lazily on the
    first run.  This is a problem if you want to load_npz() the
    parameters of the model, since not all parameters are yet created.
    """
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            dummy_batch = (model.xp.random.random(
                (1, 3, 224, 224)) * 255).astype(model.xp.float32)
            model(dummy_batch)


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

    net = Seq()

    stage1 = Seq(
        conv=L.Convolution2D(
            3, channels[0], ksize=3, stride=2, pad=1, nobias=True),
        bn=BatchNormalization(axis=(0, 2, 3)),
        relu=F.relu,
        max_pool=fun.partial(
            F.max_pooling_2d, ksize=3, stride=2, pad=1, cover_all=False))
    net.add("stage1", stage1)

    # Stage2, Stage3, Stage4.
    for i_stage, stage_repeat in enumerate(stage_repeats):
        out_channels = channels[1 + i_stage]
        stage = ShuffleNetV2Stage(out_channels, stage_repeat)
        net.add("stage{}".format(i_stage + 2), stage)

    # Conv5
    net.add("stage5", _1x1_conv_bn_relu(channels[-1]))

    run_dummy(net)
    return net


def ShuffleNetV2(k, out_size):
    """Create a ShuffleNet v2 network that outputs a vector of size
    out_size. It consists in global average pooling + fully connected
    layer on top of the feature network above.

    """
    # We do not flatten to ease reuse of features.
    net = Seq(
        features=ShuffleNetV2Features(k),
        head=Seq(
            # Global pooling (compute mean of each channel). We leave
            # keepdims=False to get a NxC array (with H and W removed and not
            # set to 1).
            global_pool=fun.partial(F.mean, axis=(2, 3)),
            fc=L.Linear(None, out_size)))
    run_dummy(net)
    return net


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
        features = net['features']
        data = np.random.random((10, 3, 123, 567)).astype(np.float32)
        ret = features(data)
        assert ret.data.shape[:2] == (10, 1024)
        data = np.random.random((10, 3, 224, 224)).astype(np.float32)
        ret = features(data)
        assert ret.data.shape == (10, 1024, 7, 7)
