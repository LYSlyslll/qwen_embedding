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

def load_error_types(paths: List[str]) -> List[Dict[str, str]]:
    for p in paths:
        if not os.path.exists(p):
            continue
        samples: List[Dict[str, str]] = []
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
                error_text = obj.get("error")
                concept_text = obj.get("concept")
                if isinstance(error_text, str) and isinstance(concept_text, str):
                    samples.append({"error": error_text, "concept": concept_text})
        if any_parsed and samples:
            return samples

    raise FileNotFoundError(f"未找到可用输入文件：{paths}")


# ---------------- 数据集 ----------------
class TextDataset(Dataset):
    def __init__(self, texts: List[Dict[str, str]], tokenizer, max_length: int = 64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        sample = self.texts[idx]
        error_text = sample["error"]
        concept_text = sample["concept"]

        error_enc = self.tokenizer(
            error_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        concept_enc = self.tokenizer(
            concept_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "error_input_ids": error_enc["input_ids"].squeeze(0),
            "error_attention_mask": error_enc["attention_mask"].squeeze(0),
            "concept_input_ids": concept_enc["input_ids"].squeeze(0),
            "concept_attention_mask": concept_enc["attention_mask"].squeeze(0),
            "error_text": error_text,
        }
        if "token_type_ids" in error_enc:
            item["error_token_type_ids"] = error_enc["token_type_ids"].squeeze(0)
        if "token_type_ids" in concept_enc:
            item["concept_token_type_ids"] = concept_enc["token_type_ids"].squeeze(0)
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
    print(f"[INFO] 输入样本数：{len(texts)}")

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
            "error_input_ids": torch.stack([it["error_input_ids"] for it in batch], dim=0),
            "error_attention_mask": torch.stack([it["error_attention_mask"] for it in batch], dim=0),
            **({"error_token_type_ids": torch.stack([it["error_token_type_ids"] for it in batch], dim=0)} if "error_token_type_ids" in batch[0] else {}),
            "concept_input_ids": torch.stack([it["concept_input_ids"] for it in batch], dim=0),
            "concept_attention_mask": torch.stack([it["concept_attention_mask"] for it in batch], dim=0),
            **({"concept_token_type_ids": torch.stack([it["concept_token_type_ids"] for it in batch], dim=0)} if "concept_token_type_ids" in batch[0] else {}),
            "error_texts": [it["error_text"] for it in batch],
        },
    )

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    # 3) 逐批写出 JSONL（节省内存）
    idx = 0
    t0 = time.time()
    with io.open(OUT_PATH, "w", encoding="utf-8") as out:
        for batch in tqdm(loader, desc="生成文本嵌入"):
            error_enc_batch = {
                "input_ids": batch["error_input_ids"],
                "attention_mask": batch["error_attention_mask"],
            }
            if "error_token_type_ids" in batch:
                error_enc_batch["token_type_ids"] = batch["error_token_type_ids"]

            concept_enc_batch = {
                "input_ids": batch["concept_input_ids"],
                "attention_mask": batch["concept_attention_mask"],
            }
            if "concept_token_type_ids" in batch:
                concept_enc_batch["token_type_ids"] = batch["concept_token_type_ids"]

            error_vecs = embedder.encode_batch(error_enc_batch, pooling=POOLING, use_fp16=USE_FP16)
            concept_vecs = embedder.encode_batch(concept_enc_batch, pooling=POOLING, use_fp16=USE_FP16)

            final_vecs = torch.cat([error_vecs, concept_vecs], dim=1)
            final_vecs = l2_normalize(final_vecs)

            for text, vec in zip(batch["error_texts"], final_vecs):
                idx += 1
                rec = {
                    "idx": idx,
                    "error": text,
                    "embedding": [float(x) for x in vec.tolist()],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # 释放显存
            del error_enc_batch, concept_enc_batch, error_vecs, concept_vecs, final_vecs
            torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"[DONE] 写出 {idx} 行 -> {OUT_PATH} ｜耗时 {dt:.2f}s")
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
