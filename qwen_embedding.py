# -*- coding: utf-8 -*-
"""
embed_error_types_hf_albert_dp.py

任务：
- 从 /home/yanghao/math/unique_error_types.jsonl（或 .txt，每行一个）抽取“错误类型”文本
- 用本地 HuggingFace + ALBERT 生成句向量（多 GPU DataParallel）
- 逐行写出到 /home/yanghao/math/error_types_embeddings.jsonl
"""

import os
import io
import json
import time
import gc
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ---------------- 环境配置 ----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cudnn.benchmark = True

# ---------------- 可调参数 ----------------
IN_PATHS = [
    "/home/liangyushan/DeepECT/2.txt",   # 优先读取 JSONL
    "/home/liangyushan/DeepECT/2.txt",     # 备选：纯文本，每行一个
]
OUT_PATH = "/home/liangyushan/concept_embeddings.jsonl"

MODEL_PATH = "/home/liangyushan/DeepECT/albert-base-chinese"

BATCH_SIZE = 4096          # 你原脚本默认；短文本 + 多卡下可很大，不够就调小
MAX_LEN    = 64            # 错误类型很短，64 足够（也能降显存）
USE_FP16   = True          # GPU 开启半精度
POOLING    = "mean"        # 可选： "mean" | "cls"
NUM_WORKERS = 8            # DataLoader 线程
PIN_MEMORY  = True

# ---------------- 数据抽取 ----------------
def _iter_llm_errors(obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    errs = obj.get("llm_analysis_error", [])
    if isinstance(errs, dict):
        errs = [errs]
    if isinstance(errs, list):
        for it in errs:
            if isinstance(it, dict):
                yield it

def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

import os
import io
import json
from typing import List

def load_error_types(paths: List[str]) -> List[str]:
    # 1) JSONL
    for p in paths:
        if not os.path.exists(p):
            continue
        if p.endswith(".jsonl"):
            all_errors = []  # 不去重，保留所有错误类型
            any_parsed = False
            with io.open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        any_parsed = True
                    except json.JSONDecodeError:
                        print(f"[WARN] 第 {i} 行 JSON 解析失败，已跳过。")
                        continue
                    if not isinstance(obj, dict):
                        continue
                    # 主字段
                    for key in ("错误类型", "error_type", "类型"):
                        if key in obj and isinstance(obj[key], str):
                            all_errors.append(obj[key])
                            break
                    # 嵌套列表
                    for err in _iter_llm_errors(obj):
                        for key in ("错误类型", "error_type", "类型"):
                            if key in err and isinstance(err[key], str):
                                all_errors.append(err[key])
                                break
            if any_parsed:
                return all_errors  # 不去重，直接返回所有数据

    # 2) 纯文本（每行一个）
    for p in paths:
        if os.path.exists(p) and p.endswith(".txt"):
            with io.open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return lines  # 不去重，直接返回所有数据

    raise FileNotFoundError(f"未找到可用输入文件：{paths}")


# ---------------- 数据集 ----------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 去掉 batch 维
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["text"] = text
        return item

# ---------------- 模型封装 ----------------
class AlbertEmbedder:
    def __init__(self, model_path: str):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"模型目录不存在：{model_path}")

        print(f"[HF] 加载本地模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.hidden_size = int(model.config.hidden_size)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # 多 GPU
        if torch.cuda.device_count() > 1:
            print(f"[INFO] 使用多 GPU：{torch.cuda.device_count()} 张（DataParallel）")
            model = DataParallel(model)
        else:
            print(f"[INFO] 使用设备：{device}")

        self.model = model
        self.device = device

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, L] -> [B, L, 1]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)          # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)                # [B, 1]
        return summed / counts

    @torch.no_grad()
    def encode_batch(self, batch: Dict[str, torch.Tensor], pooling: str = "mean", use_fp16: bool = True) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_fp16 and self.device.startswith("cuda"))):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            if pooling == "cls":
                sent = last_hidden[:, 0, :]         # [B, H]
            else:
                sent = self._mean_pool(last_hidden, attention_mask)

        sent = sent.float().cpu()
        # 兜底修复
        if torch.isnan(sent).any() or torch.isinf(sent).any():
            print("[WARN] 检测到 NaN/Inf，已用 nan_to_num 替换。")
            sent = torch.nan_to_num(sent, nan=0.0, posinf=1e9, neginf=-1e9)
        return sent

# ---------------- 工具 ----------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

# ---------------- 主流程 ----------------
def main():
    # 1) 读数据
    texts = load_error_types(IN_PATHS)
    if not texts:
        print("[WARN] 输入为空。")
        return
    print(f"[INFO] 去重后错误类型数：{len(texts)}")

    # 2) 初始化
    embedder = AlbertEmbedder(MODEL_PATH)
    print(f"[INFO] 嵌入维度 dim={embedder.hidden_size}")

    dataset = TextDataset(texts, embedder.tokenizer, max_length=MAX_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=lambda batch: {
            # 把若干样本堆叠成批
            "input_ids": torch.stack([it["input_ids"] for it in batch], dim=0),
            "attention_mask": torch.stack([it["attention_mask"] for it in batch], dim=0),
            **({"token_type_ids": torch.stack([it["token_type_ids"] for it in batch], dim=0)} if "token_type_ids" in batch[0] else {}),
            "texts": [it["text"] for it in batch],
        },
    )

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    # 3) 逐批写出 JSONL（节省内存）
    idx = 0
    t0 = time.time()
    with io.open(OUT_PATH, "w", encoding="utf-8") as out:
        for batch in tqdm(loader, desc="生成文本嵌入"):
            enc_batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            vecs = embedder.encode_batch(enc_batch, pooling=POOLING, use_fp16=USE_FP16)
            vecs = l2_normalize(vecs)  # 和你之前一致，做 L2 归一化

            for text, vec in zip(batch["texts"], vecs):
                idx += 1
                rec = {
                    "idx": idx,
                    "错误类型": text,
                    "dim": embedder.hidden_size,
                    "embedding": [float(x) for x in vec.tolist()],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # 释放显存
            del enc_batch, vecs
            torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"[DONE] 写出 {idx} 行 -> {OUT_PATH} ｜耗时 {dt:.2f}s")
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
