# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset
import re
# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive:
            num_frames = len(os.listdir(subject_path)) - 2
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)
            
            random_y = []
            
            full_range = np.arange(0, num_frames-sample_duration+1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, 'sub-'+subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            if self.input_type == 'rest':
                # ABCD rest image shape: 79, 97, 85
                # latest version might be 96,96,95
                y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject20227 = str(subject_name)+'_20227_2_0'
            subject_path = os.path.join(img_root, subject20227)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive:
                y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                background_value = rand_y.flatten()[0]
                rand_y = rand_y.permute(0,4,1,2,3)
                rand_y = torch.nn.functional.pad(rand_y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                rand_y = rand_y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex
                }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex,
                    } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()

        if self.contrastive:
            rand_y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16)
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subj,
                "target": target,
                "TR": 0,
                }
        else:
            return {
                    "fmri_sequence": y,
                    "subject_name": subj,
                    "target": target,
                    "TR": 0,
                    "sex": sex,
                    } 

class ADNIDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        # 遍历所有被试者
        for i, (subject, (sex, target)) in enumerate(subject_dict.items()):
            subject_path = os.path.join(img_root, subject)
            if target == 2:
                print('有')
            if not os.path.exists(subject_path):
                print(f"Subject path {subject_path} not found. Skipping...")
                continue

            num_frames = len([f for f in os.listdir(subject_path) if f.startswith('frame_')])  # 只统计 frame 文件
            session_duration = num_frames - self.sample_duration + 1

            # 创建时间序列
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

        # 如果是训练数据集，并且是回归任务，存储目标值的均值和标准差用于标准化
        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # ============ Contrastive learning ============
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            # ---- first view ----
            background_value = y.flatten()[0]
            # [B, Z, Y, X, T] -> [B, T, Z, Y, X]（与你示例保持一致的维度顺序）
            y = y.permute(0, 4, 1, 2, 3)
            # 原始 [61,73,61] -> 目标 [96,96,96]
            # F.pad 顺序: (X_left, X_right, Y_left, Y_right, Z_left, Z_right)
            y = torch.nn.functional.pad(
                y,
                (17, 18,  # X: 61 -> 96
                 11, 12,  # Y: 73 -> 96
                 17, 18),  # Z: 61 -> 96
                value=background_value
            )
            # 回到 [B, Z, Y, X, T]
            y = y.permute(0, 2, 3, 4, 1)

            # ---- second view ----
            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0, 4, 1, 2, 3)
            rand_y = torch.nn.functional.pad(
                rand_y,
                (17, 18,  # X
                 11, 12,  # Y
                 17, 18),  # Z
                value=background_value
            )
            rand_y = rand_y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        # ============ Resting / Task ============
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            # 和上面保持同样的维度处理
            y = y.permute(0, 4, 1, 2, 3)

            # 不区分 rest / task，统一把 [61,73,61] pad 到 [96,96,96]
            y = torch.nn.functional.pad(
                y,
                (17, 18,  # X: 61 -> 96
                 11, 12,  # Y: 73 -> 96
                 17, 18),  # Z: 61 -> 96
                value=background_value
            )

            # 回到 [B, Z, Y, X, T]
            y = y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }

class PPMI_ADNIDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 每个 subject_path 缓存一次帧文件的零填充宽度，避免反复扫描
        self._pad_width_cache = {}

    def _label_from_subname(self, sub_name: str) -> int:
        """
        从 self.subject_dict 获取该 subject 的 target；
        subject_dict 形如 {'sub_070': [sex, target], ...}
        """
        meta = self.subject_dict.get(sub_name)
        return int(meta[1]) if meta is not None else -1

    def _detect_frame_pad_width(self, subject_path: str) -> int:
        """
        从目录中推断 frame_????.pt 的数字零填充宽度（如 4、5），默认 4。
        """
        if subject_path in self._pad_width_cache:
            return self._pad_width_cache[subject_path]

        pad = 4  # 默认
        try:
            files = [f for f in os.listdir(subject_path) if f.startswith("frame_") and f.endswith(".pt")]
            if files:
                m = re.search(r"frame_(\d+)\.pt", files[0])
                if m:
                    pad = len(m.group(1))
        except Exception:
            pass

        self._pad_width_cache[subject_path] = pad
        return pad

    def _frame_path(self, subject_path: str, idx: int) -> str:
        """
        根据 pad 宽度构造帧路径；若不存在，再回退到不带零填充的名字。
        """
        pad = self._detect_frame_pad_width(subject_path)
        p1 = os.path.join(subject_path, f"frame_{idx:0{pad}d}.pt")
        if os.path.exists(p1):
            return p1
        # 兜底：frame_{idx}.pt
        p2 = os.path.join(subject_path, f"frame_{idx}.pt")
        return p2

    @staticmethod
    def _ensure_1zyxt(x: torch.Tensor, fpath: str) -> torch.Tensor:
        """
        把单帧张量统一成 [1, Z, Y, X] 形状。
        允许输入：
          - [Z, Y, X]
          - [1, Z, Y, X]
          - [Z, Y, X, 1] / [1, Z, Y, X, 1]（尾部时间维为1）
        """
        if x.dim() == 3:
            # [Z, Y, X] -> [1, Z, Y, X]
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            # 可能已经是 [1, Z, Y, X]
            # 也可能是 [Z, Y, X, 1]（把最后一维压掉）
            if x.shape[-1] == 1 and x.shape[0] != 1:
                # [Z, Y, X, 1] -> [Z, Y, X]
                x = x[..., 0].unsqueeze(0)  # -> [1, Z, Y, X]
        elif x.dim() == 5:
            # [1, Z, Y, X, 1] -> [1, Z, Y, X]
            if x.shape[0] == 1 and x.shape[-1] == 1:
                x = x[:, :, :, :, 0]
            else:
                raise RuntimeError(f"Unexpected 5D frame shape {tuple(x.shape)} @ {fpath}")
        else:
            raise RuntimeError(f"Unexpected frame shape {tuple(x.shape)} @ {fpath}")
        return x

    def load_sequence(self, subject_path: str, start_frame: int, sequence_length: int, num_frames: int = None):
        """
        读取 [start_frame, start_frame+sequence_length) 的帧，统一返回 [1, Z, Y, X, T]。
        若 self.contrastive=True，则返回 (y, rand_y)；否则返回 y。
        """
        def _load_one_sequence(s: int, L: int):
            frames = []
            end_idx = s + L
            for idx in range(s, end_idx):
                fpath = self._frame_path(subject_path, idx)
                x = torch.load(fpath, map_location="cpu")
                x = self._ensure_1zyxt(x, fpath)
                frames.append(x)

            # 沿时间维堆叠 -> [1, Z, Y, X, T]
            y = torch.stack(frames, dim=4).float()
            return y

        # 主序列
        y = _load_one_sequence(start_frame, sequence_length)

        if self.contrastive:
            # 产生第二个视角
            if num_frames is None:
                num_frames = len([f for f in os.listdir(subject_path) if f.startswith("frame_") and f.endswith(".pt")])
            max_start = max(0, num_frames - sequence_length)
            alt = start_frame
            if max_start > 0:
                alt = np.random.randint(0, max_start + 1)
                if alt == start_frame and max_start >= 1:
                    alt = (start_frame + 1) if start_frame < max_start else (start_frame - 1)
            rand_y = _load_one_sequence(alt, sequence_length)
            return y, rand_y

        return y

    def _set_data(self, root, subject_dict=None):
        """
        使用 DataModule 传入的 subject_dict（仅包含当次 split 的 subject）。
        不再按编号推断标签。
        """
        if subject_dict is None:
            raise ValueError("subject_dict is None — 请从 DataModule 传入当前 split 的字典。")

        # 保存到实例，供 _label_from_subname 使用
        self.subject_dict = subject_dict

        data = []
        img_root = os.path.join(root, 'img')
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"Image root not found: {img_root}")

        # 只遍历当前 split 的 subject
        subject_list = sorted(subject_dict.keys())

        missing_folders = []
        not_enough_frames = []

        for i, subject in enumerate(subject_list):
            sex, target = subject_dict[subject]
            try:
                target = int(target)
            except Exception:
                print(f"[WARN] {subject}: invalid target={target}, skipped.")
                continue

            subject_path = os.path.join(img_root, subject)
            if not os.path.isdir(subject_path):
                missing_folders.append(subject)
                continue

            # 只统计 frame_*.pt
            num_frames = len([f for f in os.listdir(subject_path)
                              if f.startswith('frame_') and f.endswith('.pt')])
            session_duration = num_frames - self.sample_duration + 1
            if session_duration <= 0:
                not_enough_frames.append((subject, num_frames))
                continue

            # 逐起点生成样本（步幅 self.stride）
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame,
                              self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

        if missing_folders:
            print(f"[PPMI/ADNI] {len(missing_folders)} subjects missing img folder: "
                  f"{missing_folders[:10]}{' ...' if len(missing_folders) > 10 else ''}")
        if not_enough_frames:
            head = ', '.join([f"{s}({n})" for s, n in not_enough_frames[:5]])
            tail = ' ...' if len(not_enough_frames) > 5 else ''
            print(f"[PPMI/ADNI] {len(not_enough_frames)} subjects have insufficient frames: {head}{tail}")

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def _pad_or_center_crop_to_96(self, y5d: torch.Tensor, fill_val: float):
        """
        输入: y5d shape = [B, T, Z, Y, X]
        目标: 输出 [B, T, 96, 96, 96]
        规则: 先 pad 至不小于 96，再对超出的轴中心裁剪到 96。
        """
        assert y5d.dim() == 5, f"Expect 5D tensor [B,T,Z,Y,X], got {tuple(y5d.shape)}"
        B, T, Z, Y, X = y5d.shape

        def need_pad(size, tgt=96):
            return max(0, tgt - size)

        # 计算各轴的 pad 量（左右尽量均分）
        px = need_pad(X);
        plx = px // 2;
        prx = px - plx
        py = need_pad(Y);
        ply = py // 2;
        pry = py - ply
        pz = need_pad(Z);
        plz = pz // 2;
        prz = pz - plz

        # F.pad 的顺序是 (W_left, W_right, H_left, H_right, D_left, D_right)
        if px or py or pz:
            y5d = torch.nn.functional.pad(y5d, (plx, prx, ply, pry, plz, prz), value=fill_val)

        # pad 后可能 >96，做中心裁剪到 96
        _, _, Z2, Y2, X2 = y5d.shape

        def center_slice(size, tgt=96):
            if size <= tgt:
                return slice(0, size)  # 不裁剪
            start = (size - tgt) // 2
            return slice(start, start + tgt)

        sz = center_slice(Z2);
        sy = center_slice(Y2);
        sx = center_slice(X2)
        y5d = y5d[:, :, sz, sy, sx]  # [B,T,96,96,96]

        return y5d

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # ============ Contrastive learning ============
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            # ---- view 1 ----
            background_value = y.flatten()[0]
            # [B, Z, Y, X, T] -> [B, T, Z, Y, X]
            y = y.permute(0, 4, 1, 2, 3)
            # pad [61,73,61] -> [96,96,96]
            # F.pad 顺序: (X_left, X_right, Y_left, Y_right, Z_left, Z_right)
            y = self._pad_or_center_crop_to_96(y, background_value)
            # -> [B, Z, Y, X, T]
            y = y.permute(0, 2, 3, 4, 1)

            # ---- view 2 ----
            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0, 4, 1, 2, 3)
            rand_y = self._pad_or_center_crop_to_96(rand_y, background_value)
            rand_y = torch.nn.functional.pad(
                rand_y,
                (17, 18,
                 11, 12,
                 17, 18),
                value=background_value
            )
            rand_y = rand_y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        # ============ 普通训练/推理 ============
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            # [B, Z, Y, X, T] -> [B, T, Z, Y, X]
            y = y.permute(0, 4, 1, 2, 3)

            # 统一 pad 到 [96,96,96]
            y = self._pad_or_center_crop_to_96(y, background_value)

            # 回到 [B, Z, Y, X, T]
            y = y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }