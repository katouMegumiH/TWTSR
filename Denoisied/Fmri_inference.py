import os
from program.generic import JsonSaver, ClassLoader
import datetime

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

generator_param = {}
inferrence_param = {}
steps_per_epoch = 1

generator_param["type"] = "generator"
generator_param["name"] = "FmriGenerator"
generator_param["pre_post_x"] = 3
generator_param["pre_post_y"] = 3
generator_param["pre_post_z"] = 3
generator_param["pre_post_t"] = 2
generator_param['center_omission_size'] = 4

generator_param[
    "train_path"
] = "sample_data\sub-129.nii"
generator_param["batch_size"] = 50
generator_param["start_frame"] = 0
generator_param["end_frame"] =50
generator_param["total_nb_block"] = 5
generator_param["steps_per_epoch"] = steps_per_epoch


inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "fmri_inferrence"
inferrence_param[
    "model_path"
] = "sample_data\\model_epoch32.h5"

inferrence_param[
    "output_file"
] = "sample_data/inferrence/129/129_result.nii"

jobdir = "sample_data/inferrence/129"

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_infer = os.path.join(jobdir, "inferrence.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)


inferrence_class.run()
