"""
core/hard_buffer.py - nano-CodeSelfPlay 动态对抗难题缓冲池 (Dynamic Adversarial HardBuffer)

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
    nano-CodeSelfPlay 动态对抗难题缓冲池
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
        candidate_hits: Optional[int] = None,
        total_candidates: int = 4,
        solve_rate: Optional[float] = None,
        step: int = 0
    ) -> bool:
        """
        向题库注入新的突变题对。若已存在完全相同的代码对则累加测试结果更新指标，否则新增。
        统一采用 candidate-level (hits / trials) 作为核心 solve_rate 指标口径。
        """
        if candidate_hits is None:
            if solve_rate is not None:
                candidate_hits = int(round(float(solve_rate) * total_candidates))
            else:
                candidate_hits = 0
        if total_candidates <= 0 or not 0 <= candidate_hits <= total_candidates:
            raise ValueError("Require 0 <= candidate_hits <= total_candidates and positive trials")

        key = self._compute_key(f_ref_src, f_mut_src)
        if key in self.items:
            # 累加已有条目的 candidate hits 与 trials
            item = self.items[key]
            item["total_trials"] = item.get("total_trials", total_candidates) + total_candidates
            item["total_hits"] = item.get("total_hits", int(round(item["solve_rate"] * total_candidates))) + candidate_hits
            item["solve_rate"] = item["total_hits"] / max(1, item["total_trials"])
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

        calc_solve_rate = candidate_hits / total_candidates
        self.items[key] = {
            "key": key,
            "seed_id": seed_id,
            "domain": list(domain),
            "f_ref_src": f_ref_src.strip(),
            "f_mut_src": f_mut_src.strip(),
            "sparsity": float(sparsity),
            "witnesses": list(witnesses) if witnesses else [],
            "solve_rate": float(calc_solve_rate),
            "total_hits": int(candidate_hits),
            "total_trials": int(total_candidates),
            "added_step": step,
            "times_sampled": 0,
            "times_solved": 0,
            "groups_evaluated": 0,
        }
        return True

    def sample(self, k: int = 1, is_tightened: bool = False) -> List[Dict[str, Any]]:
        """
        优先加权采样 k 道题目。
        优先权重综合考虑:
          - 甜蜜点区间 (前期 0.50，中后期收紧至 0.35)
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
            curriculum_threshold = 0.35 if is_tightened else 0.50
            for key in keys:
                item = self.items[key]
                p_solve = max(0.0, min(1.0, item["solve_rate"]))
                sparsity = max(0.001, item["sparsity"])
                sampled_count = item["times_sampled"]

                # 甜蜜点课程加权采样 (Sweet-Spot Curriculum Sampling):
                if 0.0 < p_solve <= curriculum_threshold:
                    diff_factor = 2.5
                elif p_solve == 0.0:
                    diff_factor = 0.6  # 盲区题目适度采样，避免早期零梯度死锁
                else:
                    diff_factor = max(0.2, 1.0 - p_solve)

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

    def update_result(
        self,
        key: str,
        solved: Optional[bool] = None,
        candidate_hits: Optional[int] = None,
        total_candidates: int = 4
    ):
        """
        记录 Breaker 对抗做题结果，统一按照候选击中率 (hits / trials) 维护 solve_rate。
        - candidate_hits: 当前组中成功找到反例的候选数量 (0..total_candidates)
        - total_candidates: 当前组的候选总数 (如 4)
        - solved: 是否至少有 1 个候选破解 (若未提供则自动由 candidate_hits > 0 推导)
        """
        if candidate_hits is None:
            raise ValueError("candidate_hits is required; group success cannot determine candidate hit rate")
        if total_candidates <= 0 or not 0 <= candidate_hits <= total_candidates:
            raise ValueError("Invalid candidate counts")
        if solved is not None and bool(solved) != (candidate_hits > 0):
            raise ValueError("solved disagrees with candidate_hits")
        if key in self.items:
            item = self.items[key]
            item["groups_evaluated"] = item.get("groups_evaluated", 0) + 1

            item["total_trials"] = item.get("total_trials", total_candidates) + total_candidates
            item["total_hits"] = item.get("total_hits", int(round(item["solve_rate"] * total_candidates))) + candidate_hits

            is_group_solved = (candidate_hits > 0) if solved is None else solved
            if is_group_solved:
                item["times_solved"] = item.get("times_solved", 0) + 1

            item["solve_rate"] = item["total_hits"] / max(1, item["total_trials"])

    def get_stats(self) -> Dict[str, Any]:
        """返回题库当前统计信息"""
        total = len(self.items)
        if total == 0:
            return {
                "total_problems": 0,
                "avg_solve_rate": 0.0,
                "avg_sparsity": 0.0,
                "avg_group_solve_rate": 0.0,
                "unsolved_count": 0
            }

        rates = [item["solve_rate"] for item in self.items.values()]
        sparsities = [item["sparsity"] for item in self.items.values()]
        group_rates = [
            (item.get("times_solved", 0) / item["groups_evaluated"])
            for item in self.items.values() if item.get("groups_evaluated", 0) > 0
        ]
        return {
            "total_problems": total,
            "avg_solve_rate": sum(rates) / total,
            "avg_sparsity": sum(sparsities) / total,
            "avg_group_solve_rate": (sum(group_rates) / len(group_rates)) if group_rates else 0.0,
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
