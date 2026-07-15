"""Selection rules: confirmation candidate set + lexicographic selection
(§6), offline verifier selection (§9), across-seed aggregation math (§14)."""

import pytest

from actsemble.protocol.confirmation import build_candidate_set, select_policy
from actsemble.protocol.seed_report import _seed_stats, t_critical_95
from actsemble.protocol.verifier_selection import select_verifier_record


def _screen(step, rate):
    return {"step": step, "checkpoint_path": f"ckpt/step_{step:06d}.pt",
            "checkpoint_hash": f"h{step}", "success_rate": rate,
            "success_count": int(rate * 50), "wilson_ci": [0, 1]}


class TestConfirmationCandidateSet:
    def test_top5_union_within_010(self):
        history = [_screen(s, r) for s, r in
                   [(2000, 0.40), (4000, 0.60), (6000, 0.55), (8000, 0.62),
                    (10000, 0.30), (12000, 0.58), (14000, 0.61)]]
        cands = build_candidate_set(history, top_k=5, within_of_best=0.10)
        # best 0.62 -> within-0.10 threshold 0.52; top5 by rate = same set here
        assert [c["step"] for c in cands] == [4000, 6000, 8000, 12000, 14000]

    def test_within_can_extend_beyond_top5(self):
        history = [_screen(s, r) for s, r in
                   [(1, 0.50), (2, 0.51), (3, 0.52), (4, 0.53), (5, 0.54),
                    (6, 0.55), (7, 0.46)]]
        cands = build_candidate_set(history, top_k=5, within_of_best=0.10)
        # threshold 0.45: everything qualifies via the within rule
        assert [c["step"] for c in cands] == [1, 2, 3, 4, 5, 6, 7]

    def test_fewer_than_five_includes_all(self):
        history = [_screen(s, r) for s, r in [(100, 0.0), (200, 0.9), (300, 0.1)]]
        cands = build_candidate_set(history, top_k=5, within_of_best=0.10)
        assert [c["step"] for c in cands] == [100, 200, 300]

    def test_empty_history_rejected(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            build_candidate_set([], top_k=5, within_of_best=0.10)


class TestPolicyLexicographicSelection:
    @staticmethod
    def _cand(step, screening_rate, confirmation_rate):
        return {"step": step,
                "screening": _screen(step, screening_rate),
                "confirmation": {"success_rate": confirmation_rate,
                                 "checkpoint_path": f"ckpt/step_{step:06d}.pt",
                                 "checkpoint_hash": f"h{step}"}}

    def test_highest_confirmation_wins(self):
        winner = select_policy([self._cand(2000, 0.9, 0.50),
                                self._cand(4000, 0.5, 0.60)])
        assert winner["step"] == 4000

    def test_confirmation_tie_broken_by_screening(self):
        winner = select_policy([self._cand(2000, 0.55, 0.60),
                                self._cand(4000, 0.65, 0.60)])
        assert winner["step"] == 4000

    def test_complete_tie_prefers_earliest_step(self):
        winner = select_policy([self._cand(6000, 0.6, 0.6),
                                self._cand(2000, 0.6, 0.6),
                                self._cand(4000, 0.6, 0.6)])
        assert winner["step"] == 2000


class TestVerifierOfflineSelection:
    @staticmethod
    def _hist(step, ranking, balanced=0.9, loss=0.1):
        return {"step": step, "checkpoint_path": f"v/step_{step:06d}.pt",
                "evaluated_on": "validation_episodes",
                "metrics": {"pairwise_ranking_accuracy": ranking,
                            "balanced_accuracy": balanced,
                            "validation_loss": loss}}

    def test_best_primary_wins(self):
        w = select_verifier_record([self._hist(100, 0.90), self._hist(200, 0.95),
                                    self._hist(300, 0.93)])
        assert w["step"] == 200

    def test_secondary_breaks_primary_tie(self):
        w = select_verifier_record([self._hist(100, 0.95, balanced=0.90),
                                    self._hist(200, 0.95, balanced=0.94)])
        assert w["step"] == 200

    def test_complete_tie_prefers_earliest(self):
        w = select_verifier_record([self._hist(300, 0.95), self._hist(100, 0.95),
                                    self._hist(200, 0.95)])
        assert w["step"] == 100

    def test_validation_loss_primary_prefers_lower(self):
        w = select_verifier_record(
            [self._hist(100, 0.9, loss=0.30), self._hist(200, 0.9, loss=0.20)],
            primary="validation_loss", secondary="pairwise_ranking_accuracy",
        )
        assert w["step"] == 200

    def test_missing_primary_metric_rejected(self):
        with pytest.raises(KeyError, match="nope"):
            select_verifier_record([self._hist(100, 0.9)], primary="nope")


class TestSeedAggregation:
    def test_stats_across_seeds(self):
        st = _seed_stats([0.1, 0.2, 0.0])
        assert st["mean_difference"] == pytest.approx(0.1)
        assert st["std_difference"] == pytest.approx(0.1)
        assert st["positive_seeds"] == 2 and st["negative_seeds"] == 0
        assert st["zero_seeds"] == 1
        half = t_critical_95(2) * 0.1 / (3 ** 0.5)
        assert st["confidence_interval_95"] == pytest.approx([0.1 - half, 0.1 + half])
        assert "df=2" in st["ci_method"]

    def test_single_seed_has_no_ci(self):
        st = _seed_stats([0.05])
        assert st["confidence_interval_95"] is None
        assert st["num_seeds"] == 1

    def test_t_critical_lookup(self):
        assert t_critical_95(4) == 2.776
        assert t_critical_95(22) == 2.060  # next tabulated df above 22
        with pytest.raises(ValueError):
            t_critical_95(0)
