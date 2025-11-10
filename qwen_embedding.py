# -*- coding: utf-8 -*-
"""
embed_error_types_local_vllm.py  (TXT 一行一条，鲁棒版)

输入:
- /home/yanghao/math/extracted_errors.txt   # 每行一个错误类型，跳过空行/以#开头的注释

输出(原子写):
- /home/yanghao/math/error_types_embeddings_3.jsonl
  (先写到同路径 .tmp，完成后原子重命名)

特性:
- 本地 vLLM + Qwen3-Embedding，eager 路径，显存友好；
- MAX_NUM_SEQS 与 BATCH_SIZE 对齐；MAX_TOKENS 自动估计；
- 批失败→单条回退；单条失败→零向量兜底(自动探维)；
- NaN/Inf 自愈；仅做轻量清洗(空白/包裹引号/BOM/全角空格)；
- vLLM 输出结构解析更鲁棒；打印 vLLM 版本与最终维度。
"""

import os
import io
import json
import re
from typing import Any, Iterable, List, Optional

import torch

# ---------------- 环境变量（先于 vLLM 初始化） ----------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_TORCH_COMPILE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------- 配置区 ----------------
IN_PATH   = "/home/yanghao/math/extracted_errors.txt"
OUT_PATH  = "/home/yanghao/math/error_types_embeddings_3.jsonl"  # 全 ASCII，避免奇怪路径问题
MODEL_PATH = "/home/share_ssd_data/nfs-data1/muwenjing/models/Qwen3-Embedding-4B"
# MODEL_PATH = "/home/share_ssd_data/nfs-data1/muwenjing/models/Qwen3-Embedding-0.6B"

BATCH_SIZE      = 64           # 依据显存调 16/32/64/128
NORMALIZE       = True
MAX_LEN         = 2048         # 模型 max seq len
TOKENS_PER_GUESS= 64           # 估计每条文本 token 数（短语很短，留余量）
GPU_UTIL        = 0.7
SWAP_SPACE      = 8
DTYPE           = "half"       # V100 用 fp16

# 派生参数：让并发序列数与批量一致，令 max_num_batched_tokens 充分而不越界
MAX_NUM_SEQS = max(1, BATCH_SIZE)
MAX_TOKENS   = min(MAX_NUM_SEQS * MAX_LEN,
                   max(TOKENS_PER_GUESS, MAX_NUM_SEQS * TOKENS_PER_GUESS))

# 可选：兼容“错误类型: xxx”/“error_type: xxx”/“类型: xxx”这种行首前缀
_KEY_PAT = re.compile(r"^(错误类型|error_type|类型)\s*[:：=]\s*(.+?)\s*$", re.UNICODE)

# ---------------- 读取 TXT：一行一个 ----------------
def _sanitize_line(s: str) -> str:
    # 去 BOM、全角空格，首尾空白
    s = s.replace("\ufeff", "").replace("\u3000", " ").strip()
    # 兼容“错误类型: xxx”这种前缀
    m = _KEY_PAT.match(s)
    if m:
        s = m.group(2).strip()
    # 去包裹引号（仅当整行被同一引号包裹）
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def read_error_types_from_txt(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    seen = set()
    uniq: List[str] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = _sanitize_line(line)
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                seen.add(s)
                uniq.append(s)
    return uniq

# ---------------- vLLM 本地嵌入封装 ----------------
class LocalQwenEmbedder:
    def __init__(self, model_path: str):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"模型路径无效（不存在或不是目录）：{model_path}")
        try:
            import vllm
            from vllm import LLM
            self._vllm_version = getattr(vllm, "__version__", "unknown")
            self.LLM = LLM
        except Exception as e:
            raise RuntimeError("未安装 vllm，请先 `pip install vllm`") from e

        print(f"[vLLM] 版本: {self._vllm_version}")
        print(f"[vLLM] 加载模型: {model_path}")

        self.model = self.LLM(
            model=model_path,
            task="embed",
            dtype=DTYPE,
            max_model_len=MAX_LEN,
            max_num_batched_tokens=MAX_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_UTIL,
            swap_space=SWAP_SPACE,
            enforce_eager=True,     # 禁用 torch.compile/cudagraph，降低峰值
            trust_remote_code=False,
        )
        self._dim_cache: Optional[int] = None

    # 兼容多种输出结构
    def _extract_one_embedding(self, o: Any) -> Optional[List[float]]:
        # 1) 直接是 EmbeddingOutput
        if hasattr(o, "embedding"):
            emb = getattr(o, "embedding")
            if hasattr(emb, "tolist"): return emb.tolist()
            if isinstance(emb, (list, tuple)): return list(emb)

        # 2) EmbeddingRequestOutput(outputs=...)
        if hasattr(o, "outputs"):
            out = getattr(o, "outputs")
            if hasattr(out, "embedding"):
                emb = getattr(out, "embedding")
                if hasattr(emb, "tolist"): return emb.tolist()
                if isinstance(emb, (list, tuple)): return list(emb)
            if isinstance(out, list) and out:
                first = out[0]
                if hasattr(first, "embedding"):
                    emb = getattr(first, "embedding")
                    if hasattr(emb, "tolist"): return emb.tolist()
                    if isinstance(emb, (list, tuple)): return list(emb)

        # 3) 字典兜底
        if isinstance(o, dict):
            if "embedding" in o:
                emb = o["embedding"]
                if hasattr(emb, "tolist"): return emb.tolist()
                if isinstance(emb, (list, tuple)): return list(emb)
            if "outputs" in o:
                out = o["outputs"]
                if isinstance(out, dict) and "embedding" in out:
                    emb = out["embedding"]
                    if hasattr(emb, "tolist"): return emb.tolist()
                    if isinstance(emb, (list, tuple)): return list(emb)
                if isinstance(out, list) and out:
                    first = out[0]
                    if isinstance(first, dict) and "embedding" in first:
                        emb = first["embedding"]
                        if hasattr(emb, "tolist"): return emb.tolist()
                        if isinstance(emb, (list, tuple)): return list(emb)

        # 4) 未来版本可能有 data=[{embedding: ...}] 等
        if hasattr(o, "data"):
            data = getattr(o, "data")
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict) and "embedding" in item:
                    emb = item["embedding"]
                    if hasattr(emb, "tolist"): return emb.tolist()
                    if isinstance(emb, (list, tuple)): return list(emb)

        return None

    def _infer_dim(self) -> int:
        if self._dim_cache is not None:
            return self._dim_cache
        try:
            outs = self.model.embed(["__probe__"])
            if isinstance(outs, list) and outs:
                emb = self._extract_one_embedding(outs[0])
                if emb is not None:
                    self._dim_cache = len(emb)
                    return self._dim_cache
        except Exception as e:
            print(f"[WARN] 维度探测失败：{e}")
        # 安全兜底（常见 1024/1536），给 1024
        self._dim_cache = 1024
        return self._dim_cache

    def _tensorize(self, vecs: List[List[float]]) -> torch.Tensor:
        t = torch.tensor(vecs, dtype=torch.float32)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print("[WARN] 检测到 NaN/Inf，使用 nan_to_num 进行替换。")
            t = torch.nan_to_num(t, nan=0.0, posinf=1e9, neginf=-1e9)
        return t

    def get_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        # 优先尝试整批
        try:
            outputs = self.model.embed([t.strip() for t in texts])
            vecs: List[List[float]] = []
            for o in outputs:
                emb = self._extract_one_embedding(o)
                if emb is None:
                    od = getattr(o, "__dict__", {})
                    if od:
                        emb = self._extract_one_embedding(od)
                if emb is None:
                    raise RuntimeError(f"[vLLM] 无法解析 embedding：{type(o)} -> {o}")
                vecs.append(emb)
            return self._tensorize(vecs)
        except Exception as e:
            print(f"[WARN] 批量嵌入失败({len(texts)}条)：{e}，回退为单条处理。")

        # 回退：逐条处理，个别失败用全零向量兜底（保持顺序与长度）
        dim = self._infer_dim()
        vecs: List[List[float]] = []
        for i, t in enumerate(texts):
            try:
                out = self.model.embed([t.strip()])
                emb = None
                if isinstance(out, list) and out:
                    emb = self._extract_one_embedding(out[0])
                    if emb is None:
                        od = getattr(out[0], "__dict__", {})
                        if od:
                            emb = self._extract_one_embedding(od)
                if emb is None:
                    raise RuntimeError("单条输出结构无法解析")
                vecs.append(emb)
            except Exception as e2:
                print(f"[WARN] 第 {i} 条失败：{e2} -> 使用零向量兜底")
                vecs.append([0.0] * dim)
        return self._tensorize(vecs)

# ---------------- 小工具 ----------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

def batched(arr: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(arr), n):
        yield arr[i:i + n]

# ---------------- 主流程（原子写） ----------------
def main():
    if not os.path.exists(IN_PATH):
        print(f("[ERROR] 输入文件不存在：{IN_PATH}"))
        return
    if not os.path.isdir(MODEL_PATH):
        print(f("[ERROR] 模型路径无效：{MODEL_PATH}"))
        return

    error_types = read_error_types_from_txt(IN_PATH)
    if not error_types:
        print(f"[WARN] 未从 {IN_PATH} 读取到任何错误类型。")
        return

    print(f"[INFO] 去重后错误类型数：{len(error_types)}")
    embedder = LocalQwenEmbedder(MODEL_PATH)

    out_dir = os.path.dirname(OUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = OUT_PATH + ".tmp"

    total = len(error_types)
    done = 0
    dim_captured: Optional[int] = None

    # 原子写：写 tmp -> flush+fsync -> rename
    with io.open(tmp_path, "w", encoding="utf-8") as out:
        idx = 0
        for chunk in batched(error_types, BATCH_SIZE):
            vecs = embedder.get_embeddings_batch(chunk)
            if NORMALIZE:
                vecs = l2_normalize(vecs)

            if dim_captured is None:
                dim_captured = int(vecs.size(1))
                print(f"[INFO] 嵌入维度 dim={dim_captured}")

            for label, vec in zip(chunk, vecs):
                idx += 1
                rec = {
                    "idx": idx,
                    "错误类型": label,
                    "dim": dim_captured,
                    "embedding": [float(x) for x in vec.cpu().tolist()]
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            done += len(chunk)
            print(f"[INFO] 已完成 {done}/{total}")

        out.flush()
        try:
            os.fsync(out.fileno())
        except Exception:
            pass

    os.replace(tmp_path, OUT_PATH)
    print(f"[DONE] 写出 {total} 行 -> {OUT_PATH}")

if __name__ == "__main__":
    main()
