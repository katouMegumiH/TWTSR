import os
import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import program.loss_collection as lc
from tensorflow.keras.layers import Input
from program.generic import JsonLoader
import math
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model
from packaging import version


def create_decay_callback(initial_learning_rate, epochs_drop):
    """ This is a helper function to return a configured
    learning rate decay callback. It uses passed parameters
    to define the behavior of the decay
    """

    def step_decay(epoch):
        """ learning decay callback"""

        drop = 0.5
        decrease_pow = math.floor((1 + epoch) / epochs_drop)
        lrate = initial_learning_rate * math.pow(drop, decrease_pow)

        return lrate

    return step_decay


class core_trainer:
    """通用训练器类

    该类封装了从数据生成器与网络搭建、到优化器/损失/回调初始化，再到训练与结果持久化的完整流程。

    参数说明（来自 `trainer_json_path` 的配置）：
    - output_dir: 输出目录
    - run_uid: 运行唯一标识
    - model_string: 模型名称标识字符串
    - batch_size: 训练批大小
    - steps_per_epoch: 每个 epoch 的步数（>0 时按步训练，否则按完整 __len__ 迭代器）
    - loss: 损失函数类型（参见 loss_collection.loss_selector）
    - nb_gpus: 使用的 GPU 数量（>1 时使用 MirroredStrategy）
    - period_save: 模型检查点保存周期
    - learning_rate: 学习率
    - checkpoints_dir: 检查点保存目录（缺省为 output_dir）
    - use_multiprocessing: 生成器是否使用多进程
    - caching_validation: 是否缓存验证集以避免 I/O 重复和潜在死锁
    - nb_workers: 生成器工作进程数量
    - apply_learning_decay: 是否应用分段学习率衰减
    - initial_learning_rate / epochs_drop: 学习率衰减相关参数
    - nb_times_through_data: 数据遍历次数（决定总 epoch 数）

    可将 auto_compile 设为 False 以便在超参搜索时先修改网络再手动编译。
    """
    # This is the generic trainer class
    # auto_compile can be set to False when doing an
    # hyperparameter search to allow modification of the model
    def __init__(
        self,
        generator_obj,
        test_generator_obj,
        network_obj,
        trainer_json_path,
        auto_compile=True,
    ):

        self.network_obj = network_obj
        self.local_generator = generator_obj
        self.local_test_generator = test_generator_obj

        json_obj = JsonLoader(trainer_json_path)

        # the following line is to be backward compatible in case
        # new parameter logics are added.
        json_obj.set_default("apply_learning_decay", 0)

        json_data = json_obj.json_data
        self.output_dir = json_data["output_dir"]
        self.run_uid = json_data["run_uid"]
        self.model_string = json_data["model_string"]
        self.batch_size = json_data["batch_size"]
        self.steps_per_epoch = json_data["steps_per_epoch"]
        self.loss_type = json_data["loss"]
        self.nb_gpus = json_data["nb_gpus"]
        self.period_save = json_data["period_save"]
        self.learning_rate = json_data["learning_rate"]

        if 'checkpoints_dir' in json_data.keys():
            self.checkpoints_dir = json_data["checkpoints_dir"]
        else:
            self.checkpoints_dir = self.output_dir

        if "use_multiprocessing" in json_data.keys():
            self.use_multiprocessing = json_data["use_multiprocessing"]
        else:
            self.use_multiprocessing = True

        if "caching_validation" in json_data.keys():
            self.caching_validation = json_data["caching_validation"]
        else:
            self.caching_validation = True

        self.output_model_file_path = os.path.join(
            self.output_dir,
            self.run_uid + "_" + self.model_string + "_model.h5"
        )

        if "nb_workers" in json_data.keys():
            self.workers = json_data["nb_workers"]
        else:
            self.workers = 16

        # These parameters are related to setting up the
        # behavior of learning rates
        self.apply_learning_decay = json_data["apply_learning_decay"]

        if self.apply_learning_decay == 1:
            self.initial_learning_rate = json_data["initial_learning_rate"]
            self.epochs_drop = json_data["epochs_drop"]

        self.nb_times_through_data = json_data["nb_times_through_data"]

        # Generator has to be initialized first to provide
        # input size of network
        self.initialize_generator()

        # 当使用多 GPU 时，使用 MirroredStrategy 以构建与编译分布式模型
        if self.nb_gpus > 1:
            mirrored_strategy = tensorflow.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                if auto_compile:
                    self.initialize_network()

                self.initialize_callbacks()

                self.initialize_loss()

                self.initialize_optimizer()
                if auto_compile:
                    self.compile()
        else:
            # 单 GPU/CPU 流程
            if auto_compile:
                self.initialize_network()

            self.initialize_callbacks()

            self.initialize_loss()

            self.initialize_optimizer()

            if auto_compile:
                self.compile()

    def compile(self):
        """编译模型

        使用初始化的损失函数与优化器编译 `self.local_model`。
        """
        self.local_model.compile(
            loss=self.loss, optimizer=self.optimizer
        )  # , metrics=['mae'])

    def initialize_optimizer(self):
        """初始化优化器

        目前使用 RMSprop，并从配置中读取学习率。
        """
        self.optimizer = RMSprop(lr=self.learning_rate)

    def initialize_loss(self):
        """初始化损失函数

        通过 `loss_collection.loss_selector` 根据 `loss_type` 选择损失。
        """
        self.loss = lc.loss_selector(self.loss_type)

    def initialize_callbacks(self):
        """初始化训练回调

        - ModelCheckpoint: 按 `period_save` 保存验证损失最优的模型
        - LearningRateScheduler: 可选，基于分段衰减策略
        - OnEpochEnd: 兼容 TF 2.1.0 及以下，确保生成器的 on_epoch_end 被调用
        """

        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            self.run_uid + "_" + self.model_string +
            "-{epoch:04d}-{val_loss:.4f}.h5",
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
            period=self.period_save,
        )

        # Add on epoch_end callback

        epo_end = OnEpochEnd([self.local_generator.on_epoch_end])

        if self.apply_learning_decay == 1:
            step_decay_callback = create_decay_callback(
                self.initial_learning_rate, self.epochs_drop
            )

            lrate = LearningRateScheduler(step_decay_callback)
            callbacks_list = [checkpoint, lrate]
        else:
            callbacks_list = [checkpoint]

        if version.parse(tensorflow.__version__) <= version.parse("2.1.0"):
            callbacks_list.append(epo_end)

        self.callbacks_list = callbacks_list

    def initialize_generator(self):
        """初始化与生成器相关的训练参数

        依据 `steps_per_epoch` 决定实际的总 epoch 数：
        - steps_per_epoch > 0: 以指定步数为准，遍历次数 × floor(len/steps_per_epoch)
        - 否则: 以生成器长度为一个 epoch
        """
        # 说明：
        # - len(self.local_generator) 表示“完整遍历一次生成器”需要的步数（batch 数）
        # - nb_times_through_data 表示希望完整遍历数据的次数（粗粒度控制训练时长）
        # - steps_per_epoch > 0 时，我们固定每个 epoch 只跑 steps_per_epoch 步，
        #   因此一次完整遍历需要 floor(len / steps_per_epoch) 个 epoch；
        #   为了遍历 nb_times_through_data 次，总 epoch = nb_times_through_data × floor(len/steps_per_epoch)。
        # - steps_per_epoch == 0 时，采用“整集训练”：每个 epoch 跑完整个生成器，
        #   故总 epoch = nb_times_through_data × len(generator)。
        # 注意：当 len 不能被 steps_per_epoch 整除时，floor 会丢弃尾部不足一个 epoch 的步数，
        # 若希望精确覆盖，建议令 steps_per_epoch 能整除 len(generator)，或者使用整集训练（steps_per_epoch=0）。
        if self.steps_per_epoch > 0:
            self.epochs = self.nb_times_through_data * int(
                np.floor(len(self.local_generator) / self.steps_per_epoch)
            )
        else:
            self.epochs = self.nb_times_through_data * \
                int(len(self.local_generator))

    def initialize_network(self):
        """构建模型图

        - 从生成器推断输入张量形状
        - 使用 `network_obj` 构建 Keras `Model`
        """
        local_size = self.local_generator.get_input_size()

        input_img = Input(shape=local_size)
        self.local_model = Model(input_img, self.network_obj(input_img))

    def cache_validation(self):
        """缓存验证数据到内存

        将验证生成器的全部 batch 拼接为单个数组，以减少 I/O 并避免在使用
        `keras.utils.Sequence` 作为验证集时可能发生的死锁。
        """
        # This is used to remove IO duplication,
        # leverage memory for validation and
        # avoid deadlocks that happens when
        # using keras.utils.Sequence as validation datasets
        # 内存注意事项：
        # - 该方法会将整个验证集一次性载入内存，适合验证集规模相对可控的场景；
        # - 若验证集非常大，可能造成内存压力，可考虑关闭 caching_validation。

        input_example = self.local_test_generator.__getitem__(0)
        nb_object = int(len(self.local_test_generator))

        input_shape = list(input_example[0].shape)
        nb_samples = input_shape[0]

        input_shape[0] = input_shape[0] * nb_object

        output_shape = list(input_example[1].shape)
        output_shape[0] = output_shape[0] * nb_object

        cache_input = np.zeros(shape=input_shape, dtype=input_example[0].dtype)
        cache_output = np.zeros(
            shape=output_shape, dtype=input_example[1].dtype)

        # 将每个 batch 依次拷贝到预分配的大数组中，实现一次性缓存
        for local_index in range(len(self.local_test_generator)):
            local_data = self.local_test_generator.__getitem__(local_index)
            cache_input[
                local_index * nb_samples: (local_index + 1) * nb_samples, :
            ] = local_data[0]
            cache_output[
                local_index * nb_samples: (local_index + 1) * nb_samples, :
            ] = local_data[1]

        self.local_test_generator = (cache_input, cache_output)

    def run(self):
        """启动训练流程

        - 可选地对验证集进行缓存
        - 根据 `steps_per_epoch` 分别调用 `model.fit`（步训练或整集训练）
        - 统一传入回调、多进程与 worker 配置
        """
        # we first cache the validation data
        if self.caching_validation:
            self.cache_validation()

        if self.steps_per_epoch > 0:
            # 按步训练：固定每个 epoch 的步数
            self.model_train = self.local_model.fit(
                self.local_generator,
                validation_data=self.local_test_generator,
                steps_per_epoch=self.steps_per_epoch,  # 每个 epoch 跑固定步数
                epochs=self.epochs,                    # 由 initialize_generator 计算
                max_queue_size=32,                    # 数据预取队列大小（与多进程/线程配合）
                workers=self.workers,                 # 数据加载并发 worker 数
                shuffle=False,                        # 为保持生成器时序一致，默认不打乱
                use_multiprocessing=self.use_multiprocessing,  # 是否使用多进程加载
                callbacks=self.callbacks_list,        # 回调列表（检查点、LR 衰减等）
                initial_epoch=0,
            )
        else:
            # 整集训练：每个 epoch 覆盖整个生成器
            self.model_train = self.local_model.fit(
                self.local_generator,
                validation_data=self.local_test_generator,
                epochs=self.epochs,
                max_queue_size=32,
                workers=self.workers,
                shuffle=False,
                use_multiprocessing=True,             # 这里固定为 True，与上分支保持一致行为可按需改为 self.use_multiprocessing
                callbacks=self.callbacks_list,
                initial_epoch=0,
            )

    def finalize(self):
        """训练收尾

        - 保存训练与验证损失曲线（.npy 与 .png）
        - 持久化最终模型到 `self.output_model_file_path`
        """
        draw_plot = True

        if "loss" in self.model_train.history.keys():
            loss = self.model_train.history["loss"]
            # save losses

            save_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_loss.npy"
            )
            np.save(save_loss_path, loss)
        else:
            print("Loss data was not present")
            draw_plot = False

        if "val_loss" in self.model_train.history.keys():
            val_loss = self.model_train.history["val_loss"]

            save_val_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_val_loss.npy"
            )
            np.save(save_val_loss_path, val_loss)
        else:
            print("Val. loss data was not present")
            draw_plot = False

        # save model
        self.local_model.save(self.output_model_file_path)

        print("Saved model to disk")

        if draw_plot:
            h = plt.figure()
            plt.plot(loss, label="loss " + self.run_uid)
            plt.plot(val_loss, label="val_loss " + self.run_uid)

            if self.steps_per_epoch > 0:
                plt.xlabel(
                    "number of epochs ("
                    + str(self.batch_size * self.steps_per_epoch)
                    + " samples/epochs)"
                )
            else:
                plt.xlabel(
                    "number of epochs ("
                    + str(self.batch_size * len(self.local_generator))
                    + " samples/epochs)"
                )

            plt.ylabel("training loss")
            plt.legend()
            save_hist_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_losses.png"
            )
            plt.savefig(save_hist_path)
            plt.close(h)


# This is a helper class to fix an issue in tensorflow 2.0.
# the on_epoch_end callback from sequences is not called.
class OnEpochEnd(tensorflow.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()


class transfer_trainer(core_trainer):
    # This class is used to fine-tune a pre-trained model with additional data

    def __init__(
        self, generator_obj, test_generator_obj,
        trainer_json_path, auto_compile=True,
    ):

        # self.network_obj = network_obj
        self.local_generator = generator_obj
        self.local_test_generator = test_generator_obj

        json_obj = JsonLoader(trainer_json_path)

        # the following line is to be backward compatible in case
        # new parameter logics are added.
        json_obj.set_default("apply_learning_decay", 0)

        json_data = json_obj.json_data
        self.model_path = json_data["model_path"]
        self.output_dir = json_data["output_dir"]
        self.run_uid = json_data["run_uid"]
        self.model_string = json_data["model_string"]
        self.batch_size = json_data["batch_size"]
        self.steps_per_epoch = json_data["steps_per_epoch"]
        self.loss_type = json_data["loss"]
        self.nb_gpus = json_data["nb_gpus"]
        self.period_save = json_data["period_save"]
        self.learning_rate = json_data["learning_rate"]
        self.output_model_file_path = os.path.join(
            self.output_dir,
            self.run_uid + "_" + self.model_string + "_transfer_model.h5"
        )

        if "nb_workers" in json_data.keys():
            self.workers = json_data["nb_workers"]
        else:
            self.workers = 16

        if "caching_validation" in json_data.keys():
            self.caching_validation = json_data["caching_validation"]
        else:
            self.caching_validation = True

        if "use_multiprocessing" in json_data.keys():
            self.use_multiprocessing = json_data["use_multiprocessing"]
        else:
            self.use_multiprocessing = True

        if 'checkpoints_dir' in json_data.keys():
            self.checkpoints_dir = json_data["checkpoints_dir"]
        else:
            self.checkpoints_dir = self.output_dir

        # These parameters are related to setting up the
        # behavior of learning rates
        self.apply_learning_decay = json_data["apply_learning_decay"]

        if self.apply_learning_decay == 1:
            self.initial_learning_rate = json_data["initial_learning_rate"]
            self.epochs_drop = json_data["epochs_drop"]

        self.nb_times_through_data = json_data["nb_times_through_data"]

        # Generator has to be initialized first to provide
        # input size of network
        self.initialize_generator()

        if self.nb_gpus > 1:
            mirrored_strategy = tensorflow.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                if auto_compile:
                    self.initialize_network()

                self.initialize_callbacks()

                self.initialize_loss()

                self.initialize_optimizer()
                if auto_compile:
                    self.compile()
        else:
            if auto_compile:
                self.initialize_network()

            self.initialize_callbacks()

            self.initialize_loss()

            self.initialize_optimizer()

            if auto_compile:
                self.compile()

        # For transfer learning, knowing the
        # baseline validation loss is important
        self.baseline_val_loss = self.local_model.evaluate(
            self.local_test_generator)

    def initialize_network(self):
        self.local_model = load_model(
            self.model_path
        )

    def finalize(self):
        draw_plot = True

        # save init losses
        save_loss_path = os.path.join(
            self.checkpoints_dir,
            self.run_uid + "_" + self.model_string + "init_val_loss.npy",
        )
        np.save(save_loss_path, self.baseline_val_loss)

        if "loss" in self.model_train.history.keys():
            loss = self.model_train.history["loss"]
            # save losses

            save_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_loss.npy"
            )
            np.save(save_loss_path, loss)
        else:
            print("Loss data was not present")
            draw_plot = False

        if "val_loss" in self.model_train.history.keys():
            val_loss = self.model_train.history["val_loss"]

            save_val_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_val_loss.npy"
            )
            np.save(save_val_loss_path, val_loss)
        else:
            print("Val. loss data was not present")
            draw_plot = False

        # save model
        self.local_model.save(self.output_model_file_path)

        print("Saved model to disk")

        if draw_plot:
            h = plt.figure()
            plt.plot(loss, label="loss " + self.run_uid)
            plt.plot(val_loss, label="val_loss " + self.run_uid)

            if self.steps_per_epoch > 0:
                plt.xlabel(
                    "number of epochs ("
                    + str(self.batch_size * self.steps_per_epoch)
                    + " samples/epochs)"
                )
            else:
                plt.xlabel(
                    "number of epochs ("
                    + str(self.batch_size * len(self.local_generator))
                    + " samples/epochs)"
                )

            plt.ylabel("training loss")
            plt.legend()
            save_hist_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_losses.png"
            )
            plt.savefig(save_hist_path)
            plt.close(h)
