import program as de
import sys
from shutil import copyfile
import os
from program.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

steps_per_epoch = 10

generator_test_param["type"] = "generator"
generator_test_param["name"] = "FmriGenerator"
generator_test_param["pre_post_x"] = 3
generator_test_param["pre_post_y"] = 3
generator_test_param["pre_post_z"] = 3
generator_test_param["pre_post_t"] = 1

generator_test_param[
    "train_path"
] = "sample_data\sub-129.nii"
generator_test_param["batch_size"] = 10
generator_test_param["start_frame"] = 0
generator_test_param["end_frame"] = 100
generator_test_param["total_nb_block"] = 200
generator_test_param["steps_per_epoch"] = steps_per_epoch
generator_test_param["smri_path"]="sample_data\sub-129_T1w_in_func_space.nii.gz"

generator_param["type"] = "generator"
generator_param["name"] = "FmriGenerator"
generator_param["pre_post_x"] = 3
generator_param["pre_post_y"] = 3
generator_param["pre_post_z"] = 3
generator_param["pre_post_t"] = 1
generator_param[
    "train_path"
] = "sample_data\sub-129.nii"
generator_param["batch_size"] = 10
generator_param["start_frame"] = 0
generator_param["end_frame"] = 120 
generator_param["total_nb_block"] = 500
generator_param["steps_per_epoch"] = steps_per_epoch
generator_param["smri_path"]="sample_data\sub-129_T1w_in_func_space.nii.gz"

network_param["type"] = "network"
network_param["name"] = "build_conditional_unet"

training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param["period_save"] = 150
training_param["nb_gpus"] = 0
training_param["apply_learning_decay"] = 0
training_param["nb_times_through_data"] = 3
training_param["learning_rate"] = 0.0001

training_param["loss"] = "mean_absolute_error"
training_param["model_string"] = (
    network_param["name"]
    + "_"
    + training_param["loss"]
    + "_"
    + training_param["run_uid"]
)

training_param["use_multiprocessing"] = False

jobdir = (
    "sample_data/traintest/"
    + training_param["model_string"]
    + "_"
    + run_uid
)

training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_test_generator = os.path.join(jobdir, "test_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)

train_generator = generator_obj.find_and_build()(path_generator)
test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

training_class = trainer_obj.find_and_build()(
    train_generator, test_generator, network_callback, path_training
)

training_class.run()

training_class.finalize()
