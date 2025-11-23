from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process
from tqdm import tqdm

def read_data(nifti_path, save_root, subj_name, scaling_method='z-norm', fill_zeroback=False):
    print(f"processing: {nifti_path}", flush=True)
    try:
        data_np, meta = LoadImage()(nifti_path)
    except Exception as e:
        print(f"[ERROR] failed to load {nifti_path}: {e}", flush=True)
        return

    data = torch.as_tensor(data_np)
    save_dir = os.path.join(save_root, subj_name)
    os.makedirs(save_dir, exist_ok=True)

    # 裁剪空间范围（与你原来完全一致）
    data = data[:, 14:-7, :, :]

    # 背景识别
    background = data == 0

    # 强度归一化
    if scaling_method == 'z-norm':
        valid = data[~background]
        global_mean = valid.mean()
        global_std = valid.std()
        data_temp = (data - global_mean) / (global_std + 1e-8)
    elif scaling_method == 'minmax':
        valid = data[~background]
        vmin = valid.min()
        vmax = valid.max()
        data_temp = (data - vmin) / (vmax - vmin + 1e-8)
    else:
        valid = data[~background]
        global_mean = valid.mean()
        global_std = valid.std()
        data_temp = (data - global_mean) / (global_std + 1e-8)

    # 填充背景
    data_global = torch.empty_like(data, dtype=torch.float32)
    fill_val = 0.0 if fill_zeroback else data_temp[~background].min()
    data_global[background] = fill_val
    data_global[~background] = data_temp[~background]

    # 保存每个 TR
    data_global = data_global.half()
    data_global_split = torch.split(data_global, 1, dim=3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir, f"frame_{i}.pt"))


def main():
    dataset_name = 'ADNI'
    load_root = r'G:\BaiduSyncdisk\Alzheimer\ADNI\Early_diagnosis\AD\finished'
    save_root = r'D:\Project\AD\SwiFT-main\project\data\xxx'
    scaling_method = 'z-norm'
    expected_seq_length = 1000
    max_workers = 2

    os.makedirs(os.path.join(save_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok=True)
    img_root = os.path.join(save_root, 'img')

    subjects = [d for d in os.listdir(load_root)
                if os.path.isdir(os.path.join(load_root, d)) and d.lower().startswith('sub')]

    finished_samples = set(os.listdir(img_root))
    procs = []

    # tqdm 外层进度条
    for subj_name in tqdm(sorted(subjects), desc="Preprocessing subjects", ncols=100):
        subj_dir = os.path.join(load_root, subj_name)
        cand_paths = [
            os.path.join(subj_dir, 'Filtered_4DVolume.nii'),
            os.path.join(subj_dir, 'Filtered_4DVolume.nii.gz'),
        ]
        nifti_path = next((p for p in cand_paths if os.path.exists(p)), None)
        if nifti_path is None:
            tqdm.write(f"[WARN] {subj_name}: Filtered_4DVolume.nii(.gz) not found, skip.")
            continue

        subj_save_dir = os.path.join(img_root, subj_name)
        if subj_name in finished_samples and os.path.isdir(subj_save_dir):
            try:
                n_frames = len([f for f in os.listdir(subj_save_dir) if f.endswith('.pt')])
                if n_frames >= expected_seq_length:
                    tqdm.write(f"[INFO] {subj_name} already processed ({n_frames} frames), skip.")
                    continue
            except Exception:
                pass

        p = Process(target=read_data, args=(nifti_path, img_root, subj_name, scaling_method))
        p.start()
        procs.append(p)

        if len(procs) >= max_workers:
            for pp in procs:
                pp.join()
            procs = []

    # 等待所有子进程完成
    for pp in procs:
        pp.join()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'\nTotal {round((end_time - start_time) / 60)} minutes elapsed.')