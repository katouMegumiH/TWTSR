from nilearn import image
import numpy as np
#重采样 sMRI 到 fMRI 空间


# 1. 加载图像
fmri_img = image.load_img('sample_data\sub-129.nii')  # fMRI
smri_img = image.load_img('sample_data\sub-129OAS30663_sess-d0051_T1w.nii') # sMRI

# 2. 重采样 sMRI 到 fMRI 空间
smri_resampled = image.resample_to_img(
    source_img=smri_img,
    target_img=fmri_img,
    interpolation='continuous'
)

# 3. 检查形状

# 打印原始和重采样后的形状
print("Original sMRI shape:", smri_img.shape)           
print("Resampled sMRI shape:", smri_resampled.shape)    
print("fMRI spatial shape:", fmri_img.shape[:3])        

# 检查 affine 是否对齐（可选）
print("sMRI affine:\n", smri_resampled.affine)
print("fMRI affine:\n", fmri_img.affine[:3, :3], fmri_img.affine[:3, 3])

# 4. 保存结果
smri_resampled.to_filename('sample_data\sub-129_T1w_in_func_space.nii.gz')