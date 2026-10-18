"""
core/hard_buffer.py - nano-SPIN 动态对抗难题缓冲池 (Dynamic Adversarial HardBuffer)

功能与设计理念:
  1. 难题保鲜与抗遗忘: 将 Mutator 生成且未被当前 Breaker 破解 (p_solve < 1.0) 的高价值隐蔽题目入库。
  2. 优先经验回放 (Prioritized Experience Replay):
     根据历史破解率 (p_solve 越低权重越高)、错误稀疏度 (sparsity 越低越隐蔽权重越高)
     以及采样疲劳衰减 (times_sampled 抑制过拟合) 进行优先加权采样。
  3. 内存去重与持久化: 支持基于代码 AST/字面的精确去重与 JSONL 导出。
"""

import hashlib
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple


class HardBuffer:
    """
    nano-SPIN 动态对抗难题缓冲池
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.items: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _compute_key(f_ref_src: str, f_mut_src: str) -> str:
        content = f"{f_ref_src.strip()}\n---\n{f_mut_src.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def add(
        self,
        f_ref_src: str,
        f_mut_src: str,
        domain: Tuple[int, int] = (0, 50),
        seed_id: str = "unknown",
        sparsity: float = 0.1,
        witnesses: Optional[List[int]] = None,
        solve_rate: float = 0.0,
        step: int = 0
    ) -> bool:
        """
        向题库注入新的突变题对。若已存在完全相同的代码对则更新指标，否则新增。
        """
        key = self._compute_key(f_ref_src, f_mut_src)
        if key in self.items:
            # 更新已有条目的 solve_rate 与 witnesses
            item = self.items[key]
            item["solve_rate"] = 0.5 * (item["solve_rate"] + solve_rate)
            if witnesses:
                existing_wit = set(item.get("witnesses", []))
                existing_wit.update(witnesses)
                item["witnesses"] = sorted(list(existing_wit))
            return False

        # 容量淘汰策略: 若超限，移除被采样最多且 solve_rate 最高的题目
        if len(self.items) >= self.capacity:
            evict_key = max(
                self.items.keys(),
                key=lambda k: (self.items[k]["solve_rate"], self.items[k]["times_sampled"])
            )
            del self.items[evict_key]

        self.items[key] = {
            "key": key,
            "seed_id": seed_id,
            "domain": list(domain),
            "f_ref_src": f_ref_src.strip(),
            "f_mut_src": f_mut_src.strip(),
            "sparsity": float(sparsity),
            "witnesses": list(witnesses) if witnesses else [],
            "solve_rate": float(solve_rate),
            "added_step": step,
            "times_sampled": 0,
            "times_solved": 0,
        }
        return True

    def sample(self, k: int = 1) -> List[Dict[str, Any]]:
        """
        优先加权采样 k 道题目。
        优先权重综合考虑:
          - 难度: (1.0 - solve_rate + 0.1)
          - 隐蔽度: 1.0 / (sparsity + 0.05)
          - 探索度衰减: 1.0 / sqrt(times_sampled + 1)
        """
        if not self.items:
            return []

        keys = list(self.items.keys())
        if len(keys) <= k:
            selected_keys = keys
        else:
            weights = []
            for key in keys:
                item = self.items[key]
                p_solve = max(0.0, min(1.0, item["solve_rate"]))
                sparsity = max(0.001, item["sparsity"])
                sampled_count = item["times_sampled"]

                diff_factor = 1.0 - p_solve + 0.15
                subtle_factor = 1.0 / (sparsity + 0.05)
                decay_factor = 1.0 / math.sqrt(sampled_count + 1)

                w = diff_factor * subtle_factor * decay_factor
                weights.append(max(1e-4, w))

            # 无放回加权采样
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            selected_indices = random.choices(range(len(keys)), weights=probs, k=k)
            # 使用 set 去重保证批次内题目多样性
            unique_indices = list(dict.fromkeys(selected_indices))
            while len(unique_indices) < min(k, len(keys)):
                extra = random.randint(0, len(keys) - 1)
                if extra not in unique_indices:
                    unique_indices.append(extra)
            selected_keys = [keys[i] for i in unique_indices]

        results = []
        for key in selected_keys:
            item = self.items[key]
            item["times_sampled"] += 1
            results.append(item)

        return results

    def update_result(self, key: str, solved: bool):
        """记录 Breaker 做题结果"""
        if key in self.items:
            item = self.items[key]
            if solved:
                item["times_solved"] += 1
            # 平滑更新 solve_rate
            total = item["times_sampled"]
            if total > 0:
                item["solve_rate"] = item["times_solved"] / total

    def get_stats(self) -> Dict[str, Any]:
        """返回题库当前统计信息"""
        total = len(self.items)
        if total == 0:
            return {"total_problems": 0, "avg_solve_rate": 0.0, "avg_sparsity": 0.0}

        rates = [item["solve_rate"] for item in self.items.values()]
        sparsities = [item["sparsity"] for item in self.items.values()]
        return {
            "total_problems": total,
            "avg_solve_rate": sum(rates) / total,
            "avg_sparsity": sum(sparsities) / total,
            "unsolved_count": sum(1 for r in rates if r == 0.0),
        }

    def save_jsonl(self, path: str):
        """将题库保存至 JSONL 文件"""
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items.values():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str):
        """从 JSONL 文件恢复题库"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                key = data.get("key", self._compute_key(data["f_ref_src"], data["f_mut_src"]))
                self.items[key] = data

    def __len__(self) -> int:
        return len(self.items)
