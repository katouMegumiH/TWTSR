# Class to generate data for training 
import numpy as np
import json
import h5py
import os
import tensorflow.keras as keras
from program.generic import JsonLoader
import tifffile
import nibabel as nib
import s3fs
import glob


class MaxRetryException(Exception):
    # This is helper class for EmGenerator
    pass

# 所有数据生成器的基类，定义了通用接口
class DeepGenerator(keras.utils.Sequence):
    """
    This class instantiante the basic Generator Sequence object
    from which all Deep Interpolation generator should be generated.

    Parameters:
    json_path: a path to the json file used to parametrize the generator

    Returns:
    None
    """

    # 加载json文件
    def __init__(self, json_path):
        local_json_loader = JsonLoader(json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.local_mean = 1
        self.local_std = 1

    # 获取输入数据形式
    def get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]

        return local_obj.shape[1:]

    # 输入数据形式
    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]

        return local_obj.shape[1:]

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return [np.array([]), np.array([])]

    def __get_norm_parameters__(self, idx):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample

        Parameters:
        idx index of the sample

        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std

        return local_mean, local_std



        np.random.shuffle(self.list_samples)



# Fmri数据生成器
# NIfTI格式的fMRI数据（*.nii）4D fMRI数据（x,y,z，t）
# 输入：以目标体素为中心的时空块体（尺寸由 pre_post_x/y/z/t 定义），中心体素置零
# 输出：中心体素值（或整个中心切片）
class FmriGenerator(DeepGenerator):
    def __init__(self, json_path):
        super().__init__(json_path)

        print("fmrigenerator")

        self.raw_data_file = self.json_data["train_path"]   # 数据路径
        self.batch_size = self.json_data["batch_size"]      # 一次处理的样本量
        # 表示在 x/y/z/t 四个维度上，前后各取多少个体素作为上下文
        self.pre_post_x = self.json_data["pre_post_x"]
        self.pre_post_y = self.json_data["pre_post_y"]
        self.pre_post_z = self.json_data["pre_post_z"]
        self.pre_post_t = self.json_data["pre_post_t"]

        self.start_frame = self.json_data["start_frame"]     
        self.end_frame = self.json_data["end_frame"]             #范围
        self.total_nb_block = self.json_data["total_nb_block"]   # 总共样本块
        self.steps_per_epoch = self.json_data["steps_per_epoch"] #每轮处理几个批次


        # 中心点设为 0，防止信息泄露
        if "center_omission_size" in self.json_data.keys(): 
            self.center_omission_size = self.json_data["center_omission_size"]
        else:
            self.center_omission_size = 1

        # 输出是中心体素还是空间块在某一个时间点的数据
        if "single_voxel_output_single" in self.json_data.keys():
            self.single_voxel_output_single = self.json_data[
                "single_voxel_output_single"
            ]
        else:
            self.single_voxel_output_single = True

        # 初始化样本索引列表
        if "initialize_list" in self.json_data.keys():
            self.initialize_list = self.json_data["initialize_list"]
        else:
            self.initialize_list = 1

        # We load the entire data as it fits into memory
        # 使用 nibabel 库读取 .nii 或 .nii.gz 文件；
        #  get_fdata() 返回 numpy 数组格式的数据
        self.raw_data = nib.load(self.raw_data_file).get_fdata()
        self.data_shape = self.raw_data.shape

        # # 从数据中间区域取样计算归一化参数，避免边缘效应
        middle_vol = np.round(np.array(self.data_shape) / 2).astype("int")
        range_middle = np.round(np.array(self.data_shape) / 4).astype("int")

        # We take the middle of the volume
        # and time for range estimation to avoid edge effects
        local_center_data = self.raw_data[
            middle_vol[0] - range_middle[0]: middle_vol[0] + range_middle[0],
            middle_vol[1] - range_middle[1]: middle_vol[1] + range_middle[1],
            middle_vol[2] - range_middle[2]: middle_vol[2] + range_middle[2],
            middle_vol[3] - range_middle[3]: middle_vol[3] + range_middle[3],
        ]
         # 计算局部均值和标准差
        self.local_mean = np.mean(local_center_data.flatten())
        self.local_std = np.std(local_center_data.flatten())
        self.epoch_index = 0

        self.has_smri = False
        self.smri_data = None
        if "smri_path" in self.json_data and self.json_data["smri_path"]:
            smri_path = self.json_data["smri_path"]
            self.smri_data = nib.load(smri_path).get_fdata()
            if self.smri_data.shape[:3] != self.data_shape[:3]:
                raise ValueError(f"sMRI shape {self.smri_data.shape} != fMRI spatial shape {self.data_shape[:3]}. Please align/resample.")
            self.has_smri = True

            # 独立的 sMRI 归一化统计
            smri_center = np.round(np.array(self.smri_data.shape[:3]) / 2).astype("int")
            smri_range = np.round(np.array(self.smri_data.shape[:3]) / 4).astype("int")
            local_center_smri = self.smri_data[
                smri_center[0]-smri_range[0]: smri_center[0]+smri_range[0],
                smri_center[1]-smri_range[1]: smri_center[1]+smri_range[1],
                smri_center[2]-smri_range[2]: smri_center[2]+smri_range[2],
            ]
            self.smri_mean = float(np.mean(local_center_smri))
            self.smri_std = float(np.std(local_center_smri) + 1e-8)


        # 初始化样本索引列表（空间-时间坐标）
        if self.initialize_list == 1:
            self.x_list = []
            self.y_list = []
            self.z_list = []
            self.t_list = []

            filling_array = np.zeros(self.data_shape, dtype=bool)
    
            # 随机选取不重复的空间-时间点作为样本中心
            for index, value in enumerate(range(self.total_nb_block)):
                retake = True
                while retake:
                    x_local, y_local, z_local, t_local = self.get_random_xyzt()
                    retake = False
                    if filling_array[x_local, y_local, z_local, t_local]:
                        retake = True

                filling_array[x_local, y_local, z_local, t_local] = True

                self.x_list.append(x_local)
                self.y_list.append(y_local)
                self.z_list.append(z_local)
                self.t_list.append(t_local)

    # 在范围内随机生成
    def get_random_xyzt(self):
        x_center = np.random.randint(0, self.data_shape[0])
        y_center = np.random.randint(0, self.data_shape[1])
        z_center = np.random.randint(0, self.data_shape[2])
        t_center = np.random.randint(self.start_frame, self.end_frame)

        return x_center, y_center, z_center, t_center

    # 获取总批次
    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.x_list) / self.batch_size)))

    def on_epoch_end(self):
        if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
            self.epoch_index = self.epoch_index + 1
        else:
            # if we reach the end of the data, we roll over
            self.epoch_index = 0

    #获取一批次的输入输出
    def __getitem__(self, index):
        # This is to ensure we are going through the
        # entire data when steps_per_epoch<self.__len__
        #确定数据索引
        index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)


        #输入张量的最后维度需要 +1（若有 sMRI）
        last_dim = self.pre_post_t * 2 + 1 + (1 if self.has_smri else 0)
        input_full = np.zeros([
            self.batch_size,
            self.pre_post_x * 2 + 1,
            self.pre_post_y * 2 + 1,
            self.pre_post_z * 2 + 1,
            last_dim,
        ], dtype="float32")

        # # 输入输出 如果有t 输入是batch_size个四维数组
        # input_full = np.zeros(
        #     [
        #         self.batch_size,
        #         self.pre_post_x * 2 + 1,
        #         self.pre_post_y * 2 + 1,
        #         self.pre_post_z * 2 + 1,
        #         self.pre_post_t * 2 + 1,
        #     ],
        #     dtype="float32",
        # )

        if self.single_voxel_output_single:
            output_full = np.zeros(
                [self.batch_size, 1, 1, 1, 1], dtype="float32")
        else:
            output_full = np.zeros(
                [
                    self.batch_size,
                    self.pre_post_x * 2 + 1,
                    self.pre_post_y * 2 + 1,
                    self.pre_post_z * 2 + 1,
                    1,
                ],
                dtype="float32",
            )

        for batch_index, sample_index in enumerate(indexes):
            # 通过索引获取位置
            local_x = self.x_list[sample_index]
            local_y = self.y_list[sample_index]
            local_z = self.z_list[sample_index]
            local_t = self.t_list[sample_index]
            # 获取输入值，以当前位置为中心的立方
            input, output = self.__data_generation__(
                local_x, local_y, local_z, local_t)

            input_full[batch_index, :, :, :, :] = input
            output_full[batch_index, :, :, :, :] = output

        return input_full, output_full

    # 对某一个 (x,y,z,t) 位置生成对应的输入和输出
    def __data_generation__(self, local_x, local_y, local_z, local_t):
        

        last_dim = self.pre_post_t * 2 + 1 + (1 if self.has_smri else 0)
        input_full = np.zeros([
            1,
            self.pre_post_x * 2 + 1,
            self.pre_post_y * 2 + 1,
            self.pre_post_z * 2 + 1,
            last_dim,
        ], dtype="float32")     


        if self.single_voxel_output_single:
            output_full = np.zeros([1, 1, 1, 1, 1], dtype="float32")
        else:
            output_full = np.zeros(
                [
                    1,
                    self.pre_post_x * 2 + 1,
                    self.pre_post_y * 2 + 1,
                    self.pre_post_z * 2 + 1,
                    1,
                ],
                dtype="float32",
            )

        # 判断是否越界，动态调整上下文范围
        # We cap the x axis when touching the limit of the volume
        if local_x - self.pre_post_x < 0:
            pre_x = local_x    # 0
        else:
            pre_x = self.pre_post_x
        if local_x + self.pre_post_x > self.data_shape[0] - 1:
            post_x = self.data_shape[0] - 1 - local_x  # self.data_shape[0] 边界
        else:
            post_x = self.pre_post_x

        # We cap the y axis when touching the limit of the volume
        if local_y - self.pre_post_y < 0:
            pre_y = local_y    
        else:
            pre_y = self.pre_post_y
        if local_y + self.pre_post_y > self.data_shape[1] - 1:
            post_y = self.data_shape[1] - 1 - local_y   # 
        else:
            post_y = self.pre_post_y

        # We cap the z axis when touching the limit of the volume
        if local_z - self.pre_post_z < 0:
            pre_z = local_z
        else:
            pre_z = self.pre_post_z
        if local_z + self.pre_post_z > self.data_shape[2] - 1:
            post_z = self.data_shape[2] - 1 - local_z
        else:
            post_z = self.pre_post_z

        # We cap the t axis when touching the limit of the volume
        if local_t - self.pre_post_t < 0:
            pre_t = local_t
        else:
            pre_t = self.pre_post_t
        if local_t + self.pre_post_t > self.data_shape[3] - 1:
            post_t = self.data_shape[3] - 1 - local_t
        else:
            post_t = self.pre_post_t



        # fMRI [:, :, :, :, 0:time_window]
        input_full[0,
                (self.pre_post_x - pre_x):(self.pre_post_x + post_x + 1),
                (self.pre_post_y - pre_y):(self.pre_post_y + post_y + 1),
                (self.pre_post_z - pre_z):(self.pre_post_z + post_z + 1),
                (self.pre_post_t - pre_t): (self.pre_post_t + post_t + 1),
                ] = self.raw_data[
                             (local_x - pre_x):(local_x + post_x + 1),
                             (local_y - pre_y):(local_y + post_y + 1),
                             (local_z - pre_z):(local_z + post_z + 1),
                             (local_t - pre_t):(local_t + post_t + 1),
                    ]

        # 若存在 sMRI，写到最后一个通道 [:, :, :, :, -1]
        if self.has_smri:
            input_full[0,
                     (self.pre_post_x - pre_x):(self.pre_post_x + post_x + 1),
                     (self.pre_post_y - pre_y):(self.pre_post_y + post_y + 1),
                     (self.pre_post_z - pre_z):(self.pre_post_z + post_z + 1),
                     -1,
            ] = self.smri_data[
                        (local_x - pre_x):(local_x + post_x + 1),
                        (local_y - pre_y):(local_y + post_y + 1),
                        (local_z - pre_z):(local_z + post_z + 1),
                    ]

        # 若 single_voxel_output_single=True：输出仅为输入块中心点的值。
        if self.single_voxel_output_single:
            output_full[0, 0, 0, 0, 0] = input_full[
                0, self.pre_post_x, self.pre_post_y,
                self.pre_post_z, self.pre_post_t
            ]  #output存中间值
        # 否则：输出为输入块在某个时间点的完整空间块
        else: 
            output_full[0, :, :, :, 0] = input_full[0,
                                                    :, :, :, self.pre_post_t]
        
        input_full[
            0, self.pre_post_x, self.pre_post_y,
            self.pre_post_z, self.pre_post_t
        ] = 0

        # center_omission_size > 1，则将中心点附近的体素设置为 0（模拟缺失
        if self.center_omission_size > 1:
            local_hole = self.center_omission_size - 1
            input_full[
                0,
                (self.pre_post_x - local_hole): (self.pre_post_x + local_hole),
                (self.pre_post_y - local_hole): (self.pre_post_y + local_hole),
                (self.pre_post_z - local_hole): (self.pre_post_z + local_hole),
                self.pre_post_t,
            ] = 0


        # 归一化：fMRI 通道 vs sMRI 通道
        fmri_part = input_full[..., 0:(self.pre_post_t * 2 + 1)]
        fmri_part = (fmri_part.astype("float32") - self.local_mean) / (self.local_std + 1e-8)
        input_full[..., 0:(self.pre_post_t * 2 + 1)] = fmri_part

        if self.has_smri:
            smri_chan = input_full[..., -1]
            smri_chan = (smri_chan.astype("float32") - self.smri_mean) / self.smri_std
            input_full[..., -1] = smri_chan


        # 输出仍用 fMRI 的统计做归一化
        output_full = (output_full.astype("float32") - self.local_mean) / (self.local_std + 1e-8)

        # # 使用前面计算的 local_mean 和 local_std 进行标准化
        # input_full = (input_full.astype("float32") -
        #               self.local_mean) / self.local_std
        # output_full = (output_full.astype("float32") -
        #                self.local_mean) / self.local_std

        return input_full, output_full

