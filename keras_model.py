#
# The SELDnet architecture
#

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model
import pydot, graphviz
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np

from keras.engine import Layer
from keras import backend as K
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.layers import AveragePooling2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D, Flatten, Multiply

from keras.layers.core import Lambda

import tensorflow as tf

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

#        self.data_format = conv_utils.normalize_data_format(data_format)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling':self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate,1),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x




def se_block(input, channels, r=8):
    # Squeeze
    x = GlobalAveragePooling2D()(input)
    # Excitation
    x = Dense(channels//r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])



def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    
    # CNN
    spec_cnn = spec_start
    sad_cnn = spec_start
    doa_cnn = spec_start
    src_cnn = spec_start
    doa_only_cnn = spec_start
    
#    spec_cnn = Lambda(lambda y: y[:,:4,:,:])(spec_cnn)
#    sad_cnn = Lambda(lambda y: y[:,:4,:,:])(sad_cnn)
    
    # SAD branch
    for i, convCnt in enumerate(f_pool_size):
        sad_cnn = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(sad_cnn)
        sad_cnn = BatchNormalization()(sad_cnn)
        sad_cnn = Activation('relu')(sad_cnn)
        sad_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(sad_cnn) 
        #sad_cnn = Dropout(dropout_rate)(sad_cnn)
    
    # RNN
    sad_cnn = Permute((2, 1, 3))(sad_cnn)
    #sad_rnn = Reshape((9, -1))(sad_cnn)
    sad_rnn = Reshape((data_out[0][-2], -1))(sad_cnn)
    for nb_rnn_filt in rnn_size:
        sad_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(sad_rnn)

    
    # SRC branch
    for i, convCnt in enumerate(f_pool_size):
        src_cnn = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(src_cnn)
        src_cnn = BatchNormalization()(src_cnn)
        src_cnn = Activation('relu')(src_cnn)
        src_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(src_cnn)
        #src_cnn = Dropout(dropout_rate)(src_cnn)
        
    # RNN
    src_cnn = Permute((2, 1, 3))(src_cnn)
    src_rnn = Reshape((data_out[0][-2], -1))(src_cnn)
    for nb_rnn_filt in rnn_size:
        src_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(src_rnn)
            
            
    # DOA only  branch
    for i, convCnt in enumerate(f_pool_size):
        doa_only_cnn = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(doa_only_cnn)
        doa_only_cnn = BatchNormalization()(doa_only_cnn)
        doa_only_cnn = Activation('relu')(doa_only_cnn)
        doa_only_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(doa_only_cnn)
        #doa_only_cnn = Dropout(dropout_rate)(doa_only_cnn)
        
    # RNN
    doa_only_cnn = Permute((2, 1, 3))(doa_only_cnn)
    doa_only_rnn = Reshape((data_out[0][-2], -1))(doa_only_cnn)
    for nb_rnn_filt in rnn_size:
        doa_only_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(doa_only_rnn)
            



    # SED branch
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
    """
        if i == 0:
            spec_cnn2 = spec_cnn
        else:
            spec_cnn2 = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(spec_cnn2)
            spec_cnn2 = BatchNormalization()(spec_cnn2)
            spec_cnn2 = Activation('relu')(spec_cnn2)
            spec_cnn2 = MaxPooling2D(pool_size=(2, f_pool_size[i]))(spec_cnn2)
            if i == 1:
                spec_cnn3 = spec_cnn2
            else:
                spec_cnn3 = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(spec_cnn3)
                spec_cnn3 = BatchNormalization()(spec_cnn3)
                spec_cnn3 = Activation('relu')(spec_cnn3)
                spec_cnn3 = MaxPooling2D(pool_size=(1, f_pool_size[i]))(spec_cnn3)
        #spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn2 = BilinearUpsampling((1, 4))(spec_cnn2)
    spec_cnn3 = BilinearUpsampling((1, 2))(spec_cnn3)
    spec_cnn = Concatenate(axis=2)([spec_cnn, spec_cnn2, spec_cnn3])
    """    
    
    # ASPP module
    atrous_rates = (6, 12, 18)
    b0 = Conv2D(64, (1, 1), padding='same', use_bias=False, name='aspp0')(spec_cnn)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    # rate = 6 (12)
    b1 = SepConv_BN(spec_cnn, 64, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(spec_cnn, 64, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(spec_cnn, 64, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # Image Feature branch
    out_shape = int(np.ceil(data_in[-3] / 16))
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(spec_cnn)
    b4 = Conv2D(64, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    # concatenate ASPP branches & project
    spec_cnn = Concatenate()([b4, b0, b1, b2, b3])
    
    # RNN
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    spec_rnn = Reshape((data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
                    
            
    # DOA branch
    for i, convCnt in enumerate(f_pool_size):
        doa_cnn = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(doa_cnn)
        doa_cnn = BatchNormalization()(doa_cnn)
        doa_cnn = Activation('relu')(doa_cnn)
        doa_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(doa_cnn)
        #doa_cnn = Dropout(dropout_rate)(doa_cnn)
    
    # RNN
    doa_cnn = Permute((2, 1, 3))(doa_cnn)
    doa_rnn = Reshape((data_out[0][-2], -1))(doa_cnn)
    for nb_rnn_filt in rnn_size:
        doa_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(doa_rnn)
   


    # FC - SRC
    src = src_rnn
    for nb_fnn_filt in fnn_size:
        src = TimeDistributed(Dense(nb_fnn_filt))(src)
        src = Dropout(dropout_rate)(src)
    src1 = src
    src = TimeDistributed(Dense(data_out[2][-1]))(src)
    src = Activation('softmax', name='src_out')(src)
    
    # FC - SAD
    sad = sad_rnn
    for nb_fnn_filt in fnn_size:
        sad = TimeDistributed(Dense(nb_fnn_filt))(sad)
        sad = Dropout(dropout_rate)(sad)
    sad1 = sad
    sad = GlobalAveragePooling1D()(sad)
    #sad1 = sad
    sad = Dense(data_out[3][-1])(sad)
    sad = Activation('sigmoid', name='sad_out')(sad)    
                            
    # FC - DOA_only
    doa_only = doa_only_rnn
    for nb_fnn_filt in fnn_size:
        doa_only = TimeDistributed(Dense(nb_fnn_filt))(doa_only)
        doa_only = Dropout(dropout_rate)(doa_only)
    doa_only1 = doa_only
    doa_only = TimeDistributed(Dense(data_out[4][-1]))(doa_only)
    doa_only = Activation('tanh', name='doa_only_out')(doa_only)
    

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    #sed = Multiply(name="sed_x_sad")([sed, Activation('sigmoid')(sad1)])
    #sed = Concatenate(axis=2)([sed, src1, sad1])
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)
        
    
    # FC - DOA
    doa = doa_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    #doa = Concatenate(axis=2)([doa, doa_only1])
    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)
    doa = Multiply(name="doa_x_sed")([doa, Concatenate(axis=-1)([sed, sed, sed])])

    model = None
    if doa_objective is 'mse':
        model = Model(inputs=spec_start, outputs=[sed, doa])
        model.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse', 'mean_squared_error'], loss_weights=weights)
    elif doa_objective is 'masked_mse':
        doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
        model = Model(inputs=spec_start, outputs=[sed, doa_concat, src, sad, doa_only])
        model.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', masked_mse, 'categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error'], loss_weights=weights, metrics=['accuracy'])
    else:
        print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
        exit()
    model.summary()
    plot_model(model, to_file = "./plot_model.png")
    return model


def masked_mse(y_gt, model_out):
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :14] >= 0.5 #TODO fix this hardcoded value of number of classes
    sed_out = K.repeat_elements(sed_out, 3, -1)
    sed_out = K.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
#    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out))/keras.backend.sum(sed_out)
    return K.sum(K.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out) / K.sum(sed_out)


def pit_mse(y_gt, model_out):
    y_gt = Reshape((-1, 60, 3, 2))(y_gt)
    model_out = Reshape((-1, 60, 3, 2))(model_out)
    
    loss1 = K.mean(K.square(y_gt[:, :, :, :] - model_out[:, :, :, :]))
    loss2 = (K.mean(K.square(y_gt[:, :, :, 0] - model_out[:, :, :, 1])) + K.mean(K.square(y_gt[:, :, :, 1] - model_out[:, :, :, 0]))) / 2
    
    if K.less(loss1, loss2) == True:
        loss = loss1
    else:
        loss = loss2
    
    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return loss


# ダイス係数を計算する関数
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)
# ロス関数
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def binary_focal_loss(y_true, y_pred, alpha=0.5, gamma=0.5):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
            -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def categorical_focal_loss(y_true, y_pred, alpha=0.5, gamma=0.5):
    """
    Softmax version of focal loss.
        m
    FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
        c=1
    where m = number of classes, c = class and o = observation
    Parameters:
        alpha -- the same as weighing factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(loss, axis=1)


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse, 'pit_mse': pit_mse, 'dice_loss': dice_loss, 'binary_focal_loss': binary_focal_loss, 'categorical_focal_loss': categorical_focal_loss, 'BilinearUpsampling': BilinearUpsampling})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()



