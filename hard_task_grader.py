from __future__ import annotations

from typing import Any, Dict, List


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ordered_sequence_score(agent_seq: List[str], target_seq: List[str]) -> float:
    if not target_seq:
        return 0.0

    j = 0
    matched = 0
    for tool in agent_seq:
        if j < len(target_seq) and tool == target_seq[j]:
            matched += 1
            j += 1

    return matched / max(1, len(target_seq))


class HardTaskGrader:
    def _best_reference_sequence(self, gt: Dict[str, Any]) -> List[str]:
        optimal = gt.get("optimal_action_sequence", []) or []
        alternatives = gt.get("acceptable_alternative_sequences", []) or []
        candidates = [optimal] + [seq for seq in alternatives if seq]
        return max(candidates, key=len) if candidates else []

    def grade(self, sample: Dict[str, Any], requested_tools: List[str], scratchpad: Dict[str, Any]) -> Dict[str, Any]:
        gt = sample.get("ground_truth", {}) or {}
        weights = sample.get("grading_logic", {}) or {}

        optimal = gt.get("optimal_action_sequence", []) or []
        alternatives = gt.get("acceptable_alternative_sequences", []) or []
        references = [seq for seq in [optimal] + alternatives if seq]

        if references:
            sequence_score = max(_ordered_sequence_score(requested_tools, seq) for seq in references)
            best_ref = max(references, key=len)
        else:
            sequence_score = 0.0
            best_ref = []

        tool_correctness = len(set(optimal) & set(requested_tools)) / max(1, len(set(optimal)))

        predicted = scratchpad.get("final_action")
        actual = (gt.get("expected_final_actions") or [None])[0]

        credit_matrix = gt.get("classification_credit", {}) or {}
        final_action_score = 0.0
        if predicted is not None and actual is not None:
            final_action_score = float(
                credit_matrix.get(predicted, {}).get(actual, 0.0)
            )
            if predicted in (gt.get("expected_final_actions") or []):
                final_action_score = 1.0

        required_steps = gt.get("optimal_action_sequence", []) or []
        evidence_used = scratchpad.get("evidence_used", []) or []

        evidence_fields: List[str] = []
        for item in evidence_used:
            if isinstance(item, str):
                evidence_fields.append(item)
            elif isinstance(item, dict):
                field = item.get("field")
                if field:
                    evidence_fields.append(f"request_{field}")

        evidence_quality = len(set(required_steps) & set(evidence_fields)) / max(1, len(set(required_steps)))

        extra_tools = max(0, len(requested_tools) - len(best_ref))
        efficiency_score = max(0.0, 1.0 - (extra_tools / max(1, len(best_ref))))

        shortcut_flags = scratchpad.get("shortcut_flags", {}) or {}
        shortcut_hits = scratchpad.get("forbidden_shortcut_hits", []) or []
        shortcut_safety = 1.0 if not shortcut_hits and not any(bool(v) for v in shortcut_flags.values()) else 0.0

        w_tool = float(weights.get("tool_correctness_weight", 0.0))
        w_traj = float(weights.get("trajectory_weight", 0.0))
        w_final = float(weights.get("final_action_weight", 0.0))
        w_evidence = float(weights.get("evidence_quality_weight", 0.0))
        w_eff = float(weights.get("efficiency_weight", 0.0))
        w_shortcut = float(weights.get("shortcut_safety_weight", 0.0))  # optional
        w_sum = w_tool + w_traj + w_final + w_evidence + w_eff + w_shortcut
        if w_sum <= 0:
            w_sum = 1.0

        raw = (
            w_tool * tool_correctness
            + w_traj * sequence_score
            + w_final * final_action_score
            + w_evidence * evidence_quality
            + w_eff * efficiency_score
            + w_shortcut * shortcut_safety
        )

        final_score = _clamp01(raw / w_sum)

        return {
            "final_score": round(final_score, 4),
            "components": {
                "tool_correctness": round(tool_correctness, 4),
                "trajectory": round(sequence_score, 4),
                "final_action": round(final_action_score, 4),
                "evidence_quality": round(evidence_quality, 4),
                "efficiency": round(efficiency_score, 4),
                "shortcut_safety": round(shortcut_safety, 4),
            },
            "reference_used": best_ref,
        }