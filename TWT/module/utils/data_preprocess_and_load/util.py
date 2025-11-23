import os
import csv
import random
from collections import defaultdict

# ===== 配置 =====
CSV_PATH = r"D:\Project\AD\SwiFT-main\project\output\ADNI\misclassified_subjects\misclassified_test.csv"
SPLIT_TXT_IN  = r"D:\Project\AD\SwiFT-main\project\data\splits\ADNI\split_fixed_1.txt"  # 现有划分
SPLIT_TXT_OUT = r"D:\Project\AD\SwiFT-main\project\data\splits\ADNI\split_subjects_swapped.txt"  # 输出新划分
RANDOM_SEED = 2025

# ===== 工具函数 =====
def subj_to_int(subj: str) -> int:
    # "sub_006" -> 6
    return int(subj.split("_")[1])

def subj_class(subj: str) -> int:
    """0 = 正常人(1~160), 1 = 患者(161~320)"""
    sid = subj_to_int(subj)
    return 0 if 1 <= sid <= 160 else 1

def read_misclassified_subjects(csv_path: str) -> list:
    """读取 CSV，过滤 mode=test，收集 subject 去重"""
    subs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode", "").strip().lower() == "test":
                s = row.get("subject", "").strip()
                if s:
                    subs.append(s)
    # 去重
    subs = sorted(set(subs))
    return subs

def parse_split_txt(path: str):
    """解析你现在用的三段式 txt"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    sections = {"train_subjects": [], "val_subjects": [], "test_subjects": []}
    cur = None
    for ln in lines:
        low = ln.lower()
        if low == "train_subjects":
            cur = "train_subjects"
            continue
        if low == "val_subjects":
            cur = "val_subjects"
            continue
        if low == "test_subjects":
            cur = "test_subjects"
            continue
        if cur is None:
            # 跳过未知行
            continue
        sections[cur].append(ln)
    return sections

def write_split_txt(path: str, sections: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("train_subjects\n")
        for s in sections["train_subjects"]:
            f.write(s + "\n")
        f.write("\nval_subjects\n")
        for s in sections["val_subjects"]:
            f.write(s + "\n")
        f.write("\ntest_subjects\n")
        for s in sections["test_subjects"]:
            f.write(s + "\n")

# ===== 主逻辑 =====
def main():
    random.seed(RANDOM_SEED)

    # 1) 读 CSV（判错 subject，去重）
    bad_subs_all = read_misclassified_subjects(CSV_PATH)
    print(f"CSV 共收集到(去重后) {len(bad_subs_all)} 个判错 subject。")

    # 2) 读当前划分
    split = parse_split_txt(SPLIT_TXT_IN)
    train, val, test = split["train_subjects"], split["val_subjects"], split["test_subjects"]
    train_set, val_set, test_set = set(train), set(val), set(test)

    # 3) 只保留当前 test 中的判错 subject（有些可能来自旧 test）
    bad_in_test = sorted([s for s in bad_subs_all if s in test_set])
    print(f"当前 test 中实际存在的判错 subject 数：{len(bad_in_test)}")

    if not bad_in_test:
        print("test 中没有需要替换的判错 subject，退出。")
        return

    # 4) 统计要从 train 调换到 test 的名额（分类别）
    need_from_train = defaultdict(int)  # cls -> count
    for s in bad_in_test:
        need_from_train[subj_class(s)] += 1
    print("按类别的替换名额：", dict(need_from_train))

    # 5) 在 train 里为每个类别挑等量 subject 调到 test
    #    避免和 val/test 重复，只从 train 内选；随机但可复现
    candidates_by_cls = {0: [], 1: []}
    for s in train:
        candidates_by_cls[subj_class(s)].append(s)
    for c in candidates_by_cls.values():
        random.shuffle(c)

    picked_from_train = []  # 要从 train 换到 test 的 subject 列表
    short_cls = {}
    for cls, need in need_from_train.items():
        pool = candidates_by_cls.get(cls, [])
        take = min(need, len(pool))
        picked_from_train.extend(pool[:take])
        if take < need:
            short_cls[cls] = need - take

    if short_cls:
        print("警告：某些类别在 train 中没有足够的候选可替换：", short_cls)
        # 这里选择“放宽类别平衡”，继续尽力替换。
        # 你也可以在这里尝试从 val 里补（如需的话）：
        #   - 从 val 中同类挑人 -> 先从 val 移到 train，再从 train 移到 test
        #   - 或者直接从 val 移到 test（但会改变 val 的分布）
        # 这一步按你的需求再加。

    # 6) 执行交换：bad_in_test -> train，picked_from_train -> test
    new_train = set(train)
    new_test = set(test)

    # 移除 bad_in_test 出 test，加到 train
    for s in bad_in_test:
        if s in new_test: new_test.remove(s)
        new_train.add(s)

    # 从 train 移出挑选的，加入 test
    for s in picked_from_train:
        if s in new_train: new_train.remove(s)
        new_test.add(s)

    # 7) 写回到新文件（保持 val 不变；train/test 排个序更直观）
    new_sections = {
        "train_subjects": sorted(new_train, key=lambda x: subj_to_int(x)),
        "val_subjects":   sorted(val,      key=lambda x: subj_to_int(x)),
        "test_subjects":  sorted(new_test, key=lambda x: subj_to_int(x)),
    }
    os.makedirs(os.path.dirname(SPLIT_TXT_OUT), exist_ok=True)
    write_split_txt(SPLIT_TXT_OUT, new_sections)

    # 8) 打印一下新划分的类别统计
    def count_cls(lst):
        c0 = sum(1 for s in lst if subj_class(s) == 0)
        c1 = len(lst) - c0
        return c0, c1

    t0, t1 = count_cls(new_sections["train_subjects"])
    v0, v1 = count_cls(new_sections["val_subjects"])
    e0, e1 = count_cls(new_sections["test_subjects"])

    print("\n完成！新划分写到：", SPLIT_TXT_OUT)
    print(f"Train: {len(new_sections['train_subjects'])} (normal={t0}, patient={t1})")
    print(f"Val  : {len(new_sections['val_subjects'])} (normal={v0}, patient={v1})")
    print(f"Test : {len(new_sections['test_subjects'])} (normal={e0}, patient={e1})")

if __name__ == "__main__":
    main()