import warnings

import h5py
import numpy as np
from program.generic import JsonLoader
from tensorflow.keras.models import load_model
import program.loss_collection as lc
import nibabel as nib

class fmri_inferrence:
    # This inferrence is specific to fMRI which is raster scanning for
    # denoising

    def __init__(self, inferrence_json_path, generator_obj):
        
        # 配置参数
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj

        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.output_file = self.json_data["output_file"]
        self.model_path = self.json_data["model_path"]

        # This is used when output is a full volume to select only the center
        # currently only set to true. Future implementation could make smarter
        # scanning of the volume and leverage more
        # than just the center pixel
        # single_voxel_output_single是否输出单个体素
        if "single_voxel_output_single" in self.json_data.keys():
            self.single_voxel_output_single = self.json_data[
                "single_voxel_output_single"
            ]
        else:
            self.single_voxel_output_single = True

        self.model_path = self.json_data["model_path"]
        # 加载模型
        self.model = load_model(self.model_path)
        self.input_data_size = self.generator_obj.data_shape

    def run(self):
        # 预测结果累积到4D数组
        output_shape = tuple(self.generator_obj.data_shape)
        result_array = np.zeros(output_shape, dtype="float32")

        all_z_values = np.arange(0, self.input_data_size[2])
        all_y_values = np.arange(0, self.input_data_size[1])

        input_full = np.zeros(
            [
                all_y_values.shape[0] * all_z_values.shape[0] * self.input_data_size[3],
                self.generator_obj.pre_post_x * 2 + 1,
                self.generator_obj.pre_post_y * 2 + 1,
                self.generator_obj.pre_post_z * 2 + 1,
                self.generator_obj.pre_post_t * 2 + 1,
            ],
            dtype="float32",
        )

        for local_x in np.arange(0, self.input_data_size[0]):
            print("x=" + str(local_x))
            for index_y, local_y in enumerate(all_y_values):
                print("y=" + str(local_y))
                for index_z, local_z in enumerate(all_z_values):
                    for local_t in np.arange(0, self.input_data_size[3]):
                        (
                            input_full[
                                local_t
                                + index_z * self.input_data_size[3]
                                + index_y * self.input_data_size[3] * all_z_values.shape[0],
                                :,
                                :,
                                :,
                                :,
                            ],
                            output_tmp,
                        ) = self.generator_obj.__data_generation__(
                            local_x, local_y, local_z, local_t
                        )

            predictions_data = self.model.predict(input_full)
        
            # batchsize = 1000
            # total_samples = input_full.shape[0]
            # predictions_data = np.zeros((total_samples,) + self.model.output_shape[1:], dtype="float32")
            # for batch_start in range(0, total_samples, batchsize):
            #     batch_end = min(batch_start + batchsize, total_samples)
            #     batch_input = input_full[batch_start:batch_end]
            #     batch_predictions = self.model.predict(batch_input)
            #     predictions_data[batch_start:batch_end] = batch_predictions
            

            corrected_data = (
                predictions_data * self.generator_obj.local_std
                + self.generator_obj.local_mean
            )
            # 写入到结果数组
            for index_y, local_y in enumerate(all_y_values):
                for index_z, local_z in enumerate(all_z_values):
                    for local_t in np.arange(0, self.input_data_size[3]):
                        result_array[
                            local_x, local_y, local_z, local_t
                        ] = corrected_data[
                            local_t
                            + index_z * self.input_data_size[3]
                            + index_y * self.input_data_size[3] * all_z_values.shape[0],
                            self.generator_obj.pre_post_x,
                            self.generator_obj.pre_post_y,
                            self.generator_obj.pre_post_z,
                            :,
                        ]
        
        # 写入NIfTI文件
        output_nii = self.output_file
        if not (output_nii.endswith('.nii') or output_nii.endswith('.nii.gz')):
            output_nii += '.nii.gz'
        reference_nii = self.json_data.get('reference_nii', None)
        if reference_nii is not None:
            ref_img = nib.load(reference_nii)
            affine = ref_img.affine
            header = ref_img.header
        else:
            affine = np.eye(4)
            header = None
        nii_img = nib.Nifti1Image(result_array, affine, header)
        nib.save(nii_img, output_nii)
        print(f"NIfTI文件已保存: {output_nii}")


class core_inferrence:
    # This is the generic inferrence class
    def __init__(self, inferrence_json_path, generator_obj):
        print("core-inferrence init")
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj
        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data

        self.output_file = self.json_data["output_file"]

        if "save_raw" in self.json_data.keys():
            self.save_raw = self.json_data["save_raw"]
        else:
            self.save_raw = False

        if "rescale" in self.json_data.keys():
            self.rescale = self.json_data["rescale"]
        else:
            self.rescale = True

        self.batch_size = self.generator_obj.batch_size
        self.nb_datasets = len(self.generator_obj)
        self.indiv_shape = self.generator_obj.get_output_size()

        self.__load_model()

    def __load_model(self):
        try:
            local_model_path = self.__get_local_model_path()
            self.__load_local_model(path=local_model_path)
        except KeyError:
            self.__load_model_from_mlflow()

    def __get_local_model_path(self):
        try:
            model_path = self.json_data['model_path']
            warnings.warn('Loading model from model_path will be deprecated '
                          'in a future release')
        except KeyError:
            model_path = self.json_data['model_source']['local_path']
        return model_path

    def __load_local_model(self, path: str):
        self.model = load_model(
            path,
            custom_objects={
                "annealed_loss": lc.loss_selector("annealed_loss")},
        )

    def __load_model_from_mlflow(self):
        import mlflow

        mlflow_registry_params = \
            self.json_data['model_source']['mlflow_registry']

        model_name = mlflow_registry_params['model_name']
        model_version = mlflow_registry_params.get('model_version')
        model_stage = mlflow_registry_params.get('model_stage')

        mlflow.set_tracking_uri(mlflow_registry_params['tracking_uri'])

        if model_version is not None:
            model_uri = f"models:/{model_name}/{model_version}"
        elif model_stage:
            model_uri = f"models:/{model_name}/{model_stage}"
        else:
            # Gets the latest version without any stage
            model_uri = f"models:/{model_name}/None"

        self.model = mlflow.keras.load_model(
            model_uri=model_uri
        )

    def run(self):
        print("core-inferrence run")
        #定义输出数据的形状大小
        final_shape = [self.nb_datasets * self.batch_size]
        final_shape.extend(self.indiv_shape)

        chunk_size = [1]
        chunk_size.extend(self.indiv_shape)

        with h5py.File(self.output_file, "w") as file_handle:
            #"data"：存放模型预测结果；"raw"：保存原始输入数据，用于对比或调试
            dset_out = file_handle.create_dataset(
                "data",
                shape=tuple(final_shape),
                chunks=tuple(chunk_size),
                dtype="float32",
            )

            if self.save_raw:
                raw_out = file_handle.create_dataset(
                    "raw",
                    shape=tuple(final_shape),
                    chunks=tuple(chunk_size),
                    dtype="float32",
                )
            #预测 e:\download\FmriDeep\derivatives\preproc-spm\output\sub-01\ses-perceptionTraining01\func\sub-01_ses-perceptionTraining01_task-perception_run-01_bold_preproc.nii\sub-01_ses-perceptionTraining01_task-perception_run-01_bold_preproc.nii
            for index_dataset in np.arange(0, self.nb_datasets, 1):
                local_data = self.generator_obj.__getitem__(index_dataset)

                predictions_data = self.model.predict(local_data[0])
                #归一化
                local_mean, local_std = \
                    self.generator_obj.__get_norm_parameters__(index_dataset)
                local_size = predictions_data.shape[0]

                if self.rescale:
                    corrected_data = predictions_data * local_std + local_mean
                else:
                    corrected_data = predictions_data

                if self.save_raw:  #保存原始数据？
                    if self.rescale:
                        corrected_raw = local_data[1] * local_std + local_mean
                    else:
                        corrected_raw = local_data[1]

                    raw_out[
                        index_dataset
                        * self.batch_size:index_dataset
                        * self.batch_size
                        + local_size,
                        :,
                    ] = corrected_raw
                #写入
                start = index_dataset * self.batch_size
                end = index_dataset * self.batch_size + local_size
                dset_out[start:end, :] = corrected_data
