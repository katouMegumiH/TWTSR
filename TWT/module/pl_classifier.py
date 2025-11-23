import pytorch_lightning as pl
import os
import pickle
import monai.transforms as monai_t
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.parser import str2bool
from .utils.losses import NTXentLoss, global_local_temporal_contrastive
from .utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts
from einops import rearrange
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.classification import MulticlassAUROC
from .models.swin4d_transformer_ver7 import WindowAttention4D
from .utils.propagation import (
    compute_lag_spectrum, compute_direction_field, compute_speed_map
)


class LitClassifier(pl.LightningModule):
    def __init__(self, data_module, **kwargs):
        super().__init__()
        # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)
        self.save_hyperparameters(kwargs)

        # you should define target_values at the Dataset classes
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax':
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler

        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)

        # Heads
        if not self.hparams.pretraining:
            if (
                self.hparams.downstream_task == 'sex'
                or self.hparams.downstream_task_type == 'classification'
                or self.hparams.scalability_check
            ):
                self.output_head = load_model("clf_mlp", self.hparams)
            elif (
                self.hparams.downstream_task == 'age'
                or self.hparams.downstream_task == 'int_total'
                or self.hparams.downstream_task == 'int_fluid'
                or self.hparams.downstream_task_type == 'regression'
            ):
                self.output_head = load_model("reg_mlp", self.hparams)
        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams)
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()

        if self.hparams.adjust_thresh:
            self.threshold = 0

        # ==== traveling ====
        self.extract_wave_metrics = bool(getattr(self.hparams, "extract_wave_metrics", False))
        self.TR_default = float(getattr(self.hparams, "TR", 2.0))

        _vs = getattr(self.hparams, "voxel_spacing", [3.0, 3.0, 3.0])
        # 统一成 tuple[float, float, float]
        self.voxel_spacing = (float(_vs[0]), float(_vs[1]), float(_vs[2]))

        self.wave_save_dir = getattr(self.hparams, "wave_save_dir", "wave_metrics")
        if self.extract_wave_metrics:
            os.makedirs(self.wave_save_dir, exist_ok=True)

    def forward(self, x):
        return self.output_head(self.model(x))

    def augment(self, img):
        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="border",
            device=device,
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(
            sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1
        )
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth])

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')

        return img

    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()

        if augment_during_training:
            fmri = self.augment(fmri)

        feature = self.model(fmri)

        # Classification task
        if (
            self.hparams.downstream_task == 'sex'
            or self.hparams.downstream_task_type == 'classification'
            or self.hparams.scalability_check
        ):
            logits = self.output_head(feature)
            target = target_value.long().squeeze()
        # Regression task
        elif (
            self.hparams.downstream_task == 'age'
            or self.hparams.downstream_task == 'int_total'
            or self.hparams.downstream_task == 'int_fluid'
            or self.hparams.downstream_task_type == 'regression'
        ):
            logits = self.output_head(feature)  # (batch,1) or tuple((batch,1), (batch,1))
            unnormalized_target = target_value.float()  # (batch,1)
            if self.hparams.label_scaling_method == 'standardization':  # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (
                    unnormalized_target - self.scaler.data_min_[0]
                ) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])

        return subj, logits, target

    def _calculate_loss(self, batch, mode):
        if self.hparams.pretraining:
            fmri, subj, target_value, tr, sex = batch.values()
            # 设备用当前模块device
            dev = self.device
            cond1 = (self.hparams.in_chans == 1 and not self.hparams.with_voxel_norm)
            assert cond1, "Wrong combination of options"
            loss = torch.tensor(0.0, device=dev)

            if self.hparams.use_contrastive:
                assert self.hparams.contrastive_type != "none", "Contrastive type not specified"

                # B, C, H, W, D, T = image shape
                y, diff_y = fmri

                batch_size = y.shape[0]
                if (len(subj) != len(tuple(subj))) and mode == 'train':
                    print('Some sub-sequences in a batch came from the same subject!')
                criterion = NTXentLoss(
                    device=dev,
                    batch_size=batch_size,
                    temperature=self.hparams.temperature,
                    use_cosine_similarity=True,
                ).to(dev)
                criterion_ll = NTXentLoss(
                    device=dev, batch_size=2, temperature=self.hparams.temperature, use_cosine_similarity=True
                ).to(dev)

                # type 1: IC
                # type 2: LL
                # type 3: IC + LL
                if self.hparams.contrastive_type in [1, 3]:
                    out_global_1 = self.output_head(self.model(self.augment(y)), "g")
                    out_global_2 = self.output_head(self.model(self.augment(diff_y)), "g")
                    ic_loss = criterion(out_global_1, out_global_2)
                    loss += ic_loss

                if self.hparams.contrastive_type in [2, 3]:
                    out_local_1 = []
                    out_local_2 = []
                    out_local_swin1 = self.model(self.augment(y))
                    out_local_swin2 = self.model(self.augment(y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    out_local_swin1 = self.model(self.augment(diff_y))
                    out_local_swin2 = self.model(self.augment(diff_y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    ll_loss = 0
                    # loop over batch size
                    for i in range(out_local_1[0].shape[0]):
                        # out_local shape should be: BS, n_local_clips, D
                        ll_loss += criterion_ll(
                            torch.stack(out_local_1, dim=1)[i],
                            torch.stack(out_local_2, dim=1)[i],
                        )
                    loss += ll_loss

                result_dict = {
                    f"{mode}_loss": loss,
                }
        else:
            subj, logits, target = self._compute_logits(
                batch, augment_during_training=self.hparams.augment_during_training
            )

            if (
                self.hparams.downstream_task == 'sex'
                or self.hparams.downstream_task_type == 'classification'
                or self.hparams.scalability_check
            ):
                loss = F.cross_entropy(logits, target)  # target: LongTensor
                # 计算准确率（top-1）
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == target).float().mean()
                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_acc": acc,
                }

            elif (
                self.hparams.downstream_task == 'age'
                or self.hparams.downstream_task == 'int_total'
                or self.hparams.downstream_task == 'int_fluid'
                or self.hparams.downstream_task_type == 'regression'
            ):
                loss = F.mse_loss(logits.squeeze(), target.squeeze())
                l1 = F.l1_loss(logits.squeeze(), target.squeeze())
                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_mse": loss,
                    f"{mode}_l1_loss": l1,
                }

        self.log_dict(
            result_dict,
            prog_bar=True,
            sync_dist=False,
            add_dataloader_idx=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def _evaluate_metrics(self, subj_array, total_out, mode: str):
        """
        三分类专用版本：
          total_out 形状为 (N, C+1)，前 C 列是各类 logits，最后一列是 target（0/1/2）。
        先对同一 subject 的 logits 取均值，再基于 subject 级别计算指标。
        记录并打印：acc、balanced accuracy、macro AUROC。
        """

        device = total_out.device
        subj_array = np.asarray(subj_array)

        # ------- 解析 logits / targets -------
        D = total_out.shape[1]
        assert D > 2, f"三分类期望 total_out 形状为 (N, C+1)，但拿到 {total_out.shape}"
        num_classes = D - 1
        logits_mat = total_out[:, :num_classes]  # (N, C)
        targets = total_out[:, -1].long()  # (N,)

        # ------- 按 subject 聚合（均值 logits；target 取第一次出现的） -------
        uniq_subj, inv = np.unique(subj_array, return_inverse=True)  # uniq_subj: (S,)
        inv_t = torch.from_numpy(inv).to(device=device, dtype=torch.long)
        S = len(uniq_subj)

        # sum / count
        sum_logits = torch.zeros(S, num_classes, device=device).index_add_(0, inv_t, logits_mat)
        cnt_logits = torch.zeros(S, 1, device=device).index_add_(
            0, inv_t, torch.ones((logits_mat.shape[0], 1), device=device)
        )
        subj_avg_logits = sum_logits / cnt_logits.clamp_min(1.0)  # (S, C)

        # 每个 subject 的 target：取第一次出现的样本的 target
        first_idx = np.zeros(S, dtype=np.int64)
        seen = set()
        for i, s in enumerate(inv):
            if s not in seen:
                first_idx[s] = i
                seen.add(s)
        subj_targets = targets[first_idx.tolist()]  # (S,)

        # ------- 计算三分类指标 -------
        probs = torch.softmax(subj_avg_logits, dim=-1)  # (S, C)
        preds = torch.argmax(probs, dim=-1)  # (S,)

        acc = (preds == subj_targets).float().mean()
        bal_acc = balanced_accuracy_score(subj_targets.cpu().numpy(), preds.cpu().numpy())

        # AUROC（macro）
        try:
            auroc_metric = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)
            auroc = auroc_metric(probs, subj_targets)
        except Exception:
            auroc = torch.tensor(float('nan'), device=device)

        # ------- log & print -------
        self.log(f"{mode}_acc", acc, sync_dist=True)
        self.log(f"{mode}_balacc", torch.tensor(bal_acc, device=device, dtype=torch.float32), sync_dist=True)
        if torch.isfinite(auroc):
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

        if torch.isfinite(auroc):
            print(
                f"[Epoch {self.current_epoch}] {mode} Subject Acc: {acc:.4f}, "
                f"BalAcc: {bal_acc:.4f}, AUROC: {float(auroc):.4f}"
            )
        else:
            print(
                f"[Epoch {self.current_epoch}] {mode} Subject Acc: {acc:.4f}, "
                f"BalAcc: {bal_acc:.4f}, AUROC: NaN"
            )

    def training_step(self, batch, batch_idx):
        """
        只计算 loss 并返回，不再做任何按 subject 的筛选或保存操作。
        """
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.hparams.pretraining:
            if dataloader_idx == 0:
                self._calculate_loss(batch, mode="valid")
            else:
                self._calculate_loss(batch, mode="test")
            return

        # traveling
        # ------- 打开 return_attn（只在需要时） -------
        need_wave = bool(self.extract_wave_metrics)
        if need_wave:
            self._set_return_attn(True)

        subj, logits, target = self._compute_logits(batch)

        # ------- 立即抓 attention 并计算指标，随后关闭 -------
        if need_wave:
            tr_val = batch.get("tr", None)
            TR = float(tr_val.item()) if (tr_val is not None and hasattr(tr_val, "item")) else self.TR_default

            wave = self._extract_wave_from_last(TR=TR, voxel_spacing=self.voxel_spacing)
            self._set_return_attn(False)

            if wave is not None and hasattr(self.trainer, "is_global_zero") and self.trainer.is_global_zero:
                os.makedirs(self.wave_save_dir, exist_ok=True)
                for s in subj:  # 该 batch 里的每个 subject 都各存一份
                    out_path = os.path.join(self.wave_save_dir, f"sub_{s}.pt")
                    torch.save({"subject": s, "wave": wave}, out_path)
        # traveling

        # ---------- 3 分类 ----------
        mode = "valid" if dataloader_idx == 0 else "test"

        # softmax 概率 + argmax 预测
        probs = torch.softmax(logits, dim=1)  # (B,3)
        preds = torch.argmax(probs, dim=1)  # (B,)
        acc = (preds == target.long()).float().mean()  # batch-level acc

        self.log(
            f"{mode}_pt_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=logits.shape[0],
        )

        # 返回 (subj, [logits, target]) 供 validation_epoch_end 聚合
        # 拼接 logits 和 target： (B, C+1)
        output = torch.cat([logits, target.unsqueeze(1).float()], dim=1)

        return (subj, output.detach().cpu())

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        if not self.hparams.pretraining:
            outputs_valid = outputs[0]
            outputs_test = outputs[1]

            subj_valid, subj_test = [], []
            out_valid_list, out_test_list = [], []
            for subj, out in outputs_valid:
                subj_valid += subj
                out_valid_list.append(out)
            for subj, out in outputs_test:
                subj_test += subj
                out_test_list.append(out)

            subj_valid = np.array(subj_valid)
            subj_test = np.array(subj_test)
            total_out_valid = torch.cat(out_valid_list, dim=0)  # (Nv, C+1)
            total_out_test = torch.cat(out_test_list, dim=0)  # (Nt, C+1)

            # ------- subject 级（按被试聚合）-------
            self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
            self._evaluate_metrics(subj_test, total_out_test, mode="test")

            # ------- 新增：逐 .pt 的评估（样本级，多分类版）-------
            def _pt_metrics(total_out: torch.Tensor, mode: str):
                device = total_out.device
                D = total_out.shape[1]
                assert D > 2, f"Expect (N, C+1) for multiclass, got {total_out.shape}"

                logits_all = total_out[:, : D - 1]  # (N, C)
                targets = total_out[:, D - 1].long()  # (N,)

                preds = torch.argmax(logits_all, dim=-1)  # (N,)
                acc = (preds == targets).float().mean()

                # 平衡准确率（宏平均召回）
                from sklearn.metrics import balanced_accuracy_score

                bal = balanced_accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())

                # 多分类 AUROC（宏平均）
                try:
                    from torchmetrics.classification import MulticlassAUROC

                    probs = torch.softmax(logits_all, dim=-1)
                    num_classes = logits_all.shape[1]
                    auroc = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)(
                        probs, targets
                    )
                except Exception:
                    auroc = torch.tensor(float('nan'), device=device)

                # log + 打印
                self.log(f"{mode}_pt_acc_epoch", acc, sync_dist=True)
                self.log(
                    f"{mode}_pt_balacc_ep",
                    torch.tensor(bal, device=device, dtype=torch.float32),
                    sync_dist=True,
                )
                if torch.isfinite(auroc):
                    self.log(f"{mode}_pt_AUROC_ep", auroc, sync_dist=True)

                print(
                    f"\n[Epoch {self.current_epoch}] {mode} PT Acc: {acc:.4f}, "
                    f"BalAcc: {bal:.4f}, AUROC: "
                    f"{(float(auroc) if torch.isfinite(auroc) else float('nan')):.4f}"
                )

            _pt_metrics(total_out_valid, "valid")
            _pt_metrics(total_out_test, "test")

    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self, total_subjs, total_out, mode: str):
        """
        保存按 subject 聚合的预测结果（兼容二分类 / 多分类）。
        total_out 约定：
          - 二分类：shape (N, 2) -> [:,0]=logit, [:,1]=target
          - 多分类：shape (N, C+1) -> [:,:C]=logits, [:,-1]=target
        """
        # ---- 解析形状，取 logits 与 targets ----
        D = total_out.shape[1]
        if D == 2:
            # binary
            num_classes = 1
            logits_all = total_out[:, 0:1]  # (N,1)
            targets = total_out[:, 1].long()  # (N,)
        else:
            # multiclass
            num_classes = D - 1
            logits_all = total_out[:, :num_classes]  # (N,C)
            targets = total_out[:, -1].long()  # (N,)

        # ---- 逐 subject 聚合（把每个 subject 的所有段的 logits 做平均）----
        subj_array = np.asarray(total_subjs)
        device = total_out.device

        uniq_subj, inv = np.unique(subj_array, return_inverse=True)  # uniq_subj: (S,)
        inv_t = torch.from_numpy(inv).to(device=device, dtype=torch.long)
        S = len(uniq_subj)

        # sum logits / count
        sum_logits = torch.zeros(S, num_classes, device=device).index_add_(0, inv_t, logits_all)
        cnt_logits = torch.zeros(S, 1, device=device).index_add_(
            0, inv_t, torch.ones((logits_all.shape[0], 1), device=device)
        )
        subj_avg_logits = sum_logits / cnt_logits.clamp_min(1.0)  # (S, C)；二分类时 C=1

        # 每个 subject 的 target 取该 subject 第一次出现的那个样本
        first_idx = np.zeros(S, dtype=np.int64)
        seen = set()
        for i, s in enumerate(inv):
            if s not in seen:
                first_idx[s] = i
                seen.add(s)
        subj_targets = targets[first_idx.tolist()]  # (S,)

        # ---- 组织写入内容（与原来一致写入 txt / pkl）----
        run_id = getattr(self.hparams, "id", f"{self.hparams.project_name}_{mode}")
        out_dir = os.path.join("predictions", run_id)
        os.makedirs(out_dir, exist_ok=True)

        # 文本摘要：每行一个 subject
        txt_path = os.path.join(out_dir, f"iter_{self.current_epoch}.txt")

        # 汇总字典：便于后续分析
        summary_dict = {}

        with open(txt_path, "a+", encoding="utf-8") as f:
            for k, subj in enumerate(uniq_subj):
                avg_logit = subj_avg_logits[k]  # (C,)
                if num_classes == 1:
                    # 二分类：给出 sigmoid 概率与阈值0的预测
                    prob = torch.sigmoid(avg_logit.squeeze()).item()
                    pred = int(avg_logit.squeeze() >= 0)
                    line = (
                        f"subject:{subj} ({mode})\n"
                        f"count: {int(cnt_logits[k].item())} "
                        f"avg_logit: {avg_logit.item():.6f} prob: {prob:.6f} "
                        f"pred: {pred}  -  truth: {int(subj_targets[k].item())}\n"
                    )
                    score_to_store = prob  # 存一个标量概率以兼容旧结构
                else:
                    # 多分类：softmax 概率 + argmax 预测
                    probs = torch.softmax(avg_logit, dim=-1)  # (C,)
                    pred = int(torch.argmax(probs).item())
                    truth = int(subj_targets[k].item())
                    # 简要打印各类概率
                    prob_str = ", ".join([f"{p.item():.4f}" for p in probs])
                    line = (
                        f"subject:{subj} ({mode})\n"
                        f"count: {int(cnt_logits[k].item())} "
                        f"avg_logits: [{', '.join([f'{x.item():.6f}' for x in avg_logit])}] "
                        f"probs: [{prob_str}] "
                        f"pred: {pred}  -  truth: {truth}\n"
                    )
                    score_to_store = probs.cpu().tolist()  # 存整向量

                f.write(line)

                # 汇总到字典（兼容旧字段名）
                summary_dict[str(subj)] = {
                    "mode": mode,
                    "count": int(cnt_logits[k].item()),
                    "avg_logits": avg_logit.detach().cpu().tolist(),
                    "score": score_to_store,  # 二分类为标量，多分类为向量
                    "truth": int(subj_targets[k].item()),
                    "pred": pred,
                }

        # pkl
        pkl_path = os.path.join(out_dir, f"iter_{self.current_epoch}.pkl")
        with open(pkl_path, "wb") as fw:
            pickle.dump(summary_dict, fw)

        if getattr(self.trainer, "is_global_zero", True):
            print(f"[Epoch {self.current_epoch}] saved subject-level predictions to:")
            print(f"  {txt_path}")
            print(f"  {pkl_path}")

    def test_step(self, batch, batch_idx):
        # traveling
        need_wave = bool(self.extract_wave_metrics)
        if need_wave:
            self._set_return_attn(True)

        subj, logits, target = self._compute_logits(batch)

        if need_wave:
            tr_val = batch.get("tr", None)
            TR = float(tr_val.item()) if (tr_val is not None and hasattr(tr_val, "item")) else self.TR_default
            wave = self._extract_wave_from_last(TR=TR, voxel_spacing=self.voxel_spacing)
            self._set_return_attn(False)

            if wave is not None and hasattr(self.trainer, "is_global_zero") and self.trainer.is_global_zero:
                os.makedirs(self.wave_save_dir, exist_ok=True)
                for s in subj:
                    out_path = os.path.join(self.wave_save_dir, f"sub_{s}.pt")
                    torch.save({"subject": s, "wave": wave}, out_path)
        # traveling

        # 兼容防护：如果 logits 是 (B,) 就扩成 (B,1)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        # 按约定：拼成 (B, C+1)，最后一列是 target
        output = torch.cat([logits, target.unsqueeze(1).long()], dim=1)
        return (subj, output.detach().cpu())

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            subj_test = []
            out_test_list = []
            for subj, out in outputs:
                subj_test += subj
                out_test_list.append(out.detach())
            subj_test = np.array(subj_test)
            total_out_test = torch.cat(out_test_list, dim=0)
            self._evaluate_metrics(subj_test, total_out_test, mode="test")

    def on_train_epoch_start(self) -> None:
        # scalability_check 用的计时逻辑保留
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 200
        self.gpu_warmup = 50
        self.timings = np.zeros((self.repetitions, 1))
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx - self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx - self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx - self.gpu_warmup] = curr_time
            elif (batch_idx - self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)

                Throughput = (
                    self.repetitions
                    * self.hparams.batch_size
                    * int(self.hparams.num_nodes)
                    * int(self.hparams.devices)
                    / self.total_time
                )

                self.log(f"Throughput", Throughput, sync_dist=False)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:', mean_syn)
                print('std_syn:', std_syn)

        return super().on_train_batch_end(out, batch, batch_idx)

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")

        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = (
                self.trainer.estimated_stepping_batches
            )  # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05)  # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1

            sche = CosineAnnealingWarmUpRestarts(
                optim,
                first_cycle_steps=T_0,
                cycle_mult=T_mult,
                max_lr=base_lr,
                min_lr=1e-9,
                warmup_steps=warmup,
                gamma=gamma,
            )
            print('total iterations:', self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    def on_train_epoch_end(self):
        """
        现在不再做任何基于预测结果的筛选/保存，只保留父类逻辑。
        """
        return super().on_train_epoch_end()

    def _set_return_attn(self, flag: bool):
        for m in self.model.modules():
            if isinstance(m, WindowAttention4D):
                m.return_attn = flag
                if not flag:
                    m.last_attn = None
                    m.last_meta = None

    @torch.no_grad()
    def _extract_wave_from_last(self, TR: float, voxel_spacing):
        """
        提取旅行波特征：
          1. 全局统计指标（兼容旧实现）
          2. 全局坐标级矢量场（positions_mm / vectors / speeds）
        """

        target = None
        for m in self.model.modules():
            if isinstance(m, WindowAttention4D) and getattr(m, "last_attn", None) is not None:
                target = m
        if target is None or target.last_attn is None or target.last_meta is None:
            return None

        attn = target.last_attn  # [B_win, heads, N, N]
        meta = target.last_meta
        rel_t = target.relative_time_index  # [N, N]
        rel_xyz = target.relative_xyz  # [3, N, N]

        # --- Step 1: 基本全局指标（保持原逻辑） ---
        T_w = meta["window_size"][3]
        p_delta = compute_lag_spectrum(attn, rel_t, T_w, reduce_heads=True)  # [B_win, bins]
        dir_vec = compute_direction_field(attn, rel_t, rel_xyz, positive_only=True)  # [B_win, H, N, 3]
        dir_vec = dir_vec.mean(dim=1)  # [B_win, N, 3]，对 heads 平均

        spd = compute_speed_map(attn, rel_t, rel_xyz, voxel_spacing, TR, True, True)  # [B_win, N]
        dir_norm = dir_vec / (torch.norm(dir_vec, dim=-1, keepdim=True) + 1e-8)
        R_dir_consistency = torch.mean(torch.norm(torch.mean(dir_norm, dim=1), dim=-1))  # 聚合方向一致性
        mean_speed = torch.mean(spd)
        median_speed = torch.median(spd)
        std_speed = torch.std(spd)
        p_delta_peak = torch.mean(torch.argmax(p_delta, dim=1).float()) / T_w

        results = {
            "R_dir_consistency": float(R_dir_consistency.cpu()),
            "mean_speed": float(mean_speed.cpu()),
            "median_speed": float(median_speed.cpu()),
            "std_speed": float(std_speed.cpu()),
            "p_delta_peak": float(p_delta_peak.cpu()),
            "p_delta_len": int(p_delta.shape[-1]),
        }

        # --- Step 2: 如果有全局坐标信息，就额外保存 ---
        win_starts = getattr(target, "last_win_starts", None)  # [B_win, 3]
        rel_offsets = getattr(target, "last_rel_offsets", None)  # [N, 3]

        if (win_starts is not None) and (rel_offsets is not None):
            B_win, N = dir_vec.shape[:2]
            rel = rel_offsets.to(dir_vec.device)  # [N,3]
            starts = win_starts.to(dir_vec.device)  # [B_win,3]

            # (d,h,w) 全局体素索引 = 起点 + 相对偏移
            glob_dhw = starts[:, None, :] + rel[None, :, :]  # [B_win, N, 3]

            # voxel -> mm (x=w, y=h, z=d)
            vs = torch.tensor(voxel_spacing, device=dir_vec.device).view(1, 1, 3)
            x_mm = glob_dhw[..., 2] * vs[..., 2]
            y_mm = glob_dhw[..., 1] * vs[..., 1]
            z_mm = glob_dhw[..., 0] * vs[..., 0]

            positions_mm = torch.stack([x_mm, y_mm, z_mm], dim=-1)  # [B_win, N, 3]
            vectors = dir_vec.detach().cpu()  # [B_win, N, 3]
            speeds = spd.detach().cpu()  # [B_win, N]

            # 也把这些放入返回结果中
            results.update(
                {
                    "positions_mm": positions_mm.detach().cpu(),
                    "vectors": vectors,
                    "speeds": speeds,
                    "p_spectrum": p_delta.detach().cpu(),
                    "dir_shape": tuple(dir_vec.shape),
                }
            )

        else:
            print("[WARN] 当前注意力模块未保存窗口全局坐标，将仅返回聚合指标。")

        return results

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter
        )
        group = parser.add_argument_group("Default classifier")
        # training related
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument(
            "--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts"
        )
        group.add_argument(
            "--milestones",
            nargs="+",
            default=[100, 150],
            type=int,
            help="lr scheduler",
        )
        group.add_argument(
            "--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test"
        )

        # pretraining-related
        group.add_argument(
            "--use_contrastive",
            default=False,
            help="whether to use contrastive learning (specify --contrastive_type argument as well)",
        )
        group.add_argument(
            "--contrastive_type",
            default=3,
            type=int,
            help=(
                "combination of contrastive losses to use [1: Use the Instance contrastive loss function, "
                "2: Use the local-local temporal contrastive loss function, "
                "3: Use the sum of both loss functions]"
            ),
        )
        group.add_argument("--pretraining", default=False, help="whether to use pretraining")
        group.add_argument(
            "--augment_during_training",
            default=True,
            help="whether to augment input images during training",
        )
        group.add_argument(
            "--augment_only_affine", action='store_true', help="whether to only apply affine augmentation"
        )
        group.add_argument(
            "--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation"
        )
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")

        # model related
        group.add_argument(
            "--model", type=str, default="swin4d_ver7", help="which model to be used"
        )
        group.add_argument(
            "--in_chans", type=int, default=1, help="Channel size of input image"
        )
        group.add_argument(
            "--embed_dim",
            type=int,
            default=24,
            help="embedding size (recommend to use 24, 36, 48)",
        )
        group.add_argument(
            "--window_size",
            nargs="+",
            default=[4, 4, 4, 4],
            type=int,
            help="window size from the second layers",
        )
        group.add_argument(
            "--first_window_size",
            nargs="+",
            default=[2, 2, 2, 2],
            type=int,
            help="first window size",
        )
        group.add_argument(
            "--patch_size",
            nargs="+",
            default=[6, 6, 6, 1],
            type=int,
            help="patch size",
        )
        group.add_argument(
            "--depths",
            nargs="+",
            default=[2, 2, 6, 2],
            type=int,
            help="depth of layers in each stage",
        )
        group.add_argument(
            "--num_heads",
            nargs="+",
            default=[3, 6, 12, 24],
            type=int,
            help="The number of heads for each attention layer",
        )
        group.add_argument(
            "--c_multiplier",
            type=int,
            default=2,
            help="channel multiplier for Swin Transformer architecture",
        )
        group.add_argument(
            "--last_layer_full_MSA",
            type=str2bool,
            default=False,
            help="whether to use full-scale multi-head self-attention at the last layers",
        )
        group.add_argument(
            "--clf_head_version",
            type=str,
            default="v1",
            help="clf head version, v2 has a hidden layer",
        )
        group.add_argument(
            "--attn_drop_rate",
            type=float,
            default=0,
            help="dropout rate of attention layers",
        )

        # others
        group.add_argument(
            "--scalability_check", action='store_true', help="whether to check scalability"
        )
        group.add_argument(
            "--process_code",
            default=None,
            help="Slurm code/PBS code. Use this argument if you want to save process codes to your log",
        )

        # traveling
        group.add_argument(
            "--extract_wave_metrics",
            default=True,
            help="在valid/test阶段提取旅行波指标（滞后谱/方向/速度）",
        )
        group.add_argument(
            "--TR", type=float, default=2.0, help="fMRI TR(s)，若 batch 里有 tr 会覆盖"
        )
        group.add_argument(
            "--voxel_spacing",
            nargs="+",
            type=float,
            default=[3.0, 3.0, 3.0],
            help="体素尺寸(mm)：[sd, sh, sw]",
        )
        group.add_argument(
            "--wave_save_dir",
            type=str,
            default="wave_metrics",
            help="保存旅行波指标的目录",
        )

        return parser