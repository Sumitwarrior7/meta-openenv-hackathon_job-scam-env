from __future__ import annotations

from typing import Any, Dict, Iterable, List


class HardRewardEngine:
    def _flagged(self, scratchpad: Dict[str, Any], condition: str) -> bool:
        if scratchpad.get(condition, False):
            return True
        shortcut_flags = scratchpad.get("shortcut_flags", {})
        if isinstance(shortcut_flags, dict) and shortcut_flags.get(condition, False):
            return True
        return False

    def compute(self, sample: Dict[str, Any], requested_tools: List[str], scratchpad: Dict[str, Any]) -> float:
        reward = 0.0
        reward_logic = sample.get("reward_logic", {}) or {}

        for rule in reward_logic.get("dense_rewards", []):
            if self._flagged(scratchpad, rule.get("condition", "")):
                reward += float(rule.get("reward", 0.0))

        for rule in reward_logic.get("sparse_rewards", []):
            if self._flagged(scratchpad, rule.get("condition", "")):
                reward += float(rule.get("reward", 0.0))

        for rule in reward_logic.get("penalties", []):
            if self._flagged(scratchpad, rule.get("condition", "")):
                reward += float(rule.get("reward", 0.0))

        gt = sample.get("ground_truth", {}) or {}
        optimal_len = len(gt.get("optimal_action_sequence", []) or [])
        extra_tools = max(0, len(requested_tools) - optimal_len)
        reward -= extra_tools * float(reward_logic.get("efficiency_penalty_per_extra_tool", 0.0))

        forbidden_hits = scratchpad.get("forbidden_shortcut_hits", [])
        if isinstance(forbidden_hits, list):
            reward -= 0.15 * len(forbidden_hits)

        return round(reward, 4)

    def delta(
        self,
        sample: Dict[str, Any],
        requested_tools_before: List[str],
        requested_tools_after: List[str],
        scratchpad_before: Dict[str, Any],
        scratchpad_after: Dict[str, Any],
    ) -> float:
        before = self.compute(sample, requested_tools_before, scratchpad_before)
        after = self.compute(sample, requested_tools_after, scratchpad_after)
        return round(after - before, 4)