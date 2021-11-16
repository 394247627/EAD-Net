import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K
from .attention import PAM, CAM

from .blocks import residual_block
from .blocks import attention_block
from .blocks import attention_block1
# Source:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

from .carafe import CARAFE
from .config import IMAGE_ORDERING


if IMAGE_ORDERING == 'channels_first':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
elif IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x: x[:, :, :-1, :-1])(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x



def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

def bottleneck_Block(input, out_filters, strides=(1, 1), dilation=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same',
               dilation_rate=dilation, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def eadnet(height, width, channel, classes):
    input = Input(shape=(height, width, channel))

    conv1_1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)   #256 256 64
    conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)   #128, 128, 64)

    # conv2_x  1/4   128 128 256
    # conv2_1 = bottleneck_Block(conv1_2, 256, strides=(1, 1))
    conv2_2 = bottleneck_Block(conv2_1, 256, with_conv_shortcut=True)


    # conv3_x  1/8
    conv3_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_2)
    # conv3_2 = bottleneck_Block(conv3_1, 512, dilation=(2, 2))
    conv3_3 = bottleneck_Block(conv3_1, 512, dilation=(2, 2), with_conv_shortcut=True)

    # conv3_4 = bottleneck_Block(conv3_3, 512, dilation=(2, 2))      # 64, 64, 512

    # conv4_x  1/16

    conv4_1 = bottleneck_Block(conv3_3, 1024, strides=(2, 2))
    conv4_2 = bottleneck_Block(conv4_1, 1024, dilation=(5, 5), with_conv_shortcut=True)
 #64, 64, 2048

    # ATTENTION
    reduce_conv5_3 = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(conv4_2)  #64, 64, 512
    reduce_conv5_3 = BatchNormalization(axis=3)(reduce_conv5_3)
    reduce_conv5_3 = Activation('relu')(reduce_conv5_3)  #64, 64, 512

    pam = PAM()(reduce_conv5_3)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    pam = BatchNormalization(axis=3)(pam)
    pam = Activation('relu')(pam)
    pam = Dropout(0.5)(pam)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

    cam = CAM()(reduce_conv5_3)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    cam = BatchNormalization(axis=3)(cam)
    cam = Activation('relu')(cam)
    cam = Dropout(0.5)(cam)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)   #64, 64, 512

    feature_sum = add([pam, cam])
    feature_sum = Dropout(0.5)(feature_sum)
    feature_sum = Conv2d_BN(feature_sum, 512, 1)   # 64, 64, 512
    merge7 = concatenate([conv3_2, feature_sum], axis=3)   ## 64, 64, 1024
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)   ## 64, 64, 512


    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)  #128, 128, 256
    merge8 = concatenate([conv2_2, up8], axis=3)  #128, 128, 512
    conv8 = Conv2d_BN(merge8, 256, 3)
    # conv8 = Conv2d_BN(conv8, 256, 3)  #128, 128, 25

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)  #256 256 64
    merge9 = concatenate([conv1_1, up9], axis=3)   # 256 256 128
    conv9 = Conv2d_BN(merge9, 64, 3)
    # conv9 = Conv2d_BN(conv9, 64, 3)  # 256 256 64

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 64, 2)  # 512 512 64
    conv10 = Conv2d_BN(up10, 64, 3)   #512 512 64
    # conv10 = Conv2d_BN(conv10, 64, 3)  #512 512 64
    #
    conv11 = Conv2d_BN(conv10, classes, 1, use_activation=None)  #512 512 5
    #
    # conv10 = Conv2d_BN(conv9,64,3)
    # output = Conv2D(classes, (3, 3), padding='same')(conv10)
    activation = Activation('softmax', name='Classification')(conv11)

    from .model_utils import get_segmentation_model

    model = get_segmentation_model(input, activation)
    # model = Model(inputs=input, outputs=activation)
    return model
