import numpy as np
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv2D,
    Conv3D,
    MaxPooling1D,
    MaxPooling2D,
    MaxPool3D,
    UpSampling3D,
    UpSampling2D,
    Dense,
    ZeroPadding2D,
    ZeroPadding3D,
    Flatten,
    DepthwiseConv2D,
    Dropout,
    Reshape,
    GlobalAveragePooling3D
)
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.layers import dot
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from program.generic import JsonLoader

import tensorflow as tf


# 基于 U-Net 结构 的 3D 卷积网络；包含编码器（Encoder）和解码器（Decoder）；
# 使用跳跃连接Skip Connection；最后一层使用 Conv3D(1) 输出去噪后的信号
def fmri_unet_denoiser(path_json):
    def local_network_function(input_img):
        # encoder
        conv1 = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(pool2)

        # decoder
        up1 = UpSampling3D((2, 2, 2))(conv3)
        up1 = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(up1)

        conc_up_1 = Concatenate()([up1, conv2])

        conv4 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(conc_up_1)

        up2 = UpSampling3D((2, 2, 2))(conv4)
        up2 = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(up2)

        conc_up_2 = Concatenate()([up2, conv1])

        conv5 = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(conc_up_2)

        out = Conv3D(1, (1, 1, 1), activation=None, padding="same")(conv5)
        return out

    return local_network_function

# 支持自动超参数搜索（Keras Tuner）；可变层数（卷积层和全连接层）；
# 支持不同激活函数选择；编码器+解码器混合结构；最终通过全连接层回归目标信号。
def fmri_flexible_architecture(path_json):
    def local_network_function(input_img, hp):
        # encoder
        in_conv = input_img
        out_conv = input_img

        broad_activation = hp.Choice("unit_activation", values=["relu", "elu"])

        for nb_conv in range(hp.Choice(f"nb_conv_layers", values=[0, 1, 2])):
            conv_interm = Conv3D(
                hp.Choice(
                    f"conv_{nb_conv}_units", values=[32, 64, 128, 256], default=64
                ),
                (2, 2, 2),
                activation=broad_activation,
                padding="same",
            )(in_conv)
            out_conv = MaxPool3D(pool_size=(2, 2, 2))(conv_interm)
            in_conv = out_conv

        in_dense = out_conv

        for nb_dense in range(hp.Choice(f"nb_dense_layers", values=[2, 4, 6])):
            out_dense = Dense(
                hp.Choice(
                    f"dense_{nb_dense}_units", values=[32, 64, 128, 256], default=128
                ),
                activation=broad_activation,
            )(in_dense)
            in_dense = out_dense

        final = Dense(1, activation=None)(out_dense)

        return final

    return local_network_function

# 更深的密集层（Dense Layers）；使用较多 Dense 层进行特征抽象；
# 卷积层较少，强调全连接建模；最终输出单通道信号。
def fmri_volume_optimized_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(256, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(128, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        dens1 = Dense(64, activation="relu")(pool2)
        dens2 = Dense(32, activation="relu")(dens1)
        dens3 = Dense(64, activation="relu")(dens2)
        dens4 = Dense(64, activation="relu")(dens3)
        dens5 = Dense(64, activation="relu")(dens4)
        dens6 = Dense(64, activation="relu")(dens5)

        dense_out = Dense(1, activation=None)(dens6)

        return dense_out

    return local_network_function

# 较多卷积层 + 多个 Dense 层；编码器结构较深；强调多层次特征提取；适合复杂非线性关系建模
def fmri_volume_deeper_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(32, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(64, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        dens1 = Dense(128, activation="relu")(pool2)
        dens2 = Dense(128, activation="relu")(dens1)
        dens3 = Dense(128, activation="relu")(dens2)
        dens4 = Dense(128, activation="relu")(dens3)

        dense_out = Dense(1, activation=None)(dens4)

        return dense_out

    return local_network_function

# 相对简单但有效的结构；卷积层 + 少量 Dense 层；易于训练、收敛速度快；平衡性能与效率。
def fmri_volume_dense_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(32, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(64, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        dens1 = Dense(128, activation="relu")(pool2)
        dens2 = Dense(128, activation="relu")(dens1)

        dense_out = Dense(1, activation=None)(dens2)

        return dense_out

    return local_network_function

# 标准卷积+密集层组合；包含三个卷积层，两个密集层；比 dense_denoiser 略深一些；更强的局部模式识别能力。
def fmri_volume_denoiser(path_json):

    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(32, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(64, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        conv3 = Conv3D(128, (2, 2, 2), activation="relu", padding="same")(pool2)
        dens1 = Dense(128, activation="relu")(conv3)
        dens2 = Dense(128, activation="relu")(dens1)

        dense_out = Dense(1, activation=None)(dens2)

        return dense_out

    return local_network_function

#+smri
def build_conditional_unet(path_josn):

    def local_network_function(input_img):

        fmri_input = input_img[..., 0:7]
        smri_input = input_img[..., -1]
        
        # 轻量编码器，提取 sMRI 全局特征，并通过全局池化得到样本级别的条件向量
        x = smri_input
        x = tf.expand_dims(x, axis=-1) 
        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPool3D((2, 2, 2))(x)  

        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPool3D((2, 2, 2))(x)  

        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = GlobalAveragePooling3D()(x) 

        # 维度需与后续被调制层的通道数严格匹配
        gamma = Dense(16, name='gamma')(x)   # 对应 conv4 的 16 通道  全连接Dense
        beta  = Dense(16, name='beta')(x)

        gamma2 = Dense(8, name='gamma2')(x)  # 对应 conv5 的 8 通道
        beta2  = Dense(8, name='beta2')(x)

        # 调制函数
        def film_modulation(feat, gamma, beta):

            # 将样本级条件向量 gamma/beta 扩展到空间维度，并对特征图做逐通道仿射变换：
            # feat' = gamma * feat + beta
           
            # 扩展 gamma 和 beta 到空间维度: (B, C) -> (B, 1, 1, 1, C)
            gamma = gamma[:, None, None, None, :]
            beta  = beta  [:, None, None, None, :]
            return gamma * feat + beta

        # Encoder（与 fmri_unet_denoiser 相同的 3D U-Net 编码结构）
        conv1 = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(fmri_input)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)  

        conv2 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)  

        conv3 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(pool2)

        # Decoder（结构同 fmri_unet_denoiser，但在解码阶段插入 FiLM 条件调制）
        up1 = UpSampling3D((2, 2, 2))(conv3)
        up1 = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(up1)  # 补齐尺寸

        conc_up_1 = Concatenate()([up1, conv2])  # Skip connection

        conv4 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(conc_up_1)

        # FiLM 调制：用 sMRI 条件向量调节 conv4 的 16 个通道（scale+shift）
        conv4 = film_modulation(conv4, gamma, beta)

        up2 = UpSampling3D((2, 2, 2))(conv4)
        up2 = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(up2)
        conc_up_2 = Concatenate()([up2, conv1])

        conv5 = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(conc_up_2)
        # FiLM 调制：再次用 sMRI 条件向量调节 conv5 的 8 个通道
        conv5 = film_modulation(conv5, gamma2, beta2)

        # 输出层
        out = Conv3D(1, (1, 1, 1), activation=None, padding="same")(conv5)

        return out

    return local_network_function