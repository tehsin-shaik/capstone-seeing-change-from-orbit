"""Tests for the verdict line that dinov3/train_dinov3_lora.py writes into RESULTS.md.

The random-forest bar (0.704) is a mean over all 5 folds. A run of fewer folds must never be given a
yes/no against it. No weights, data or GPU are needed. Run from the code folder:
    python -m pytest -q tests/
Skipped automatically if PyTorch, transformers or peft is not installed.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

CODE = Path(__file__).resolve().parents[1]
for p in (CODE, CODE / "dinov3"):
    sys.path.insert(0, str(p))

import train_dinov3 as T  # noqa: E402
import train_dinov3_lora as L  # noqa: E402

ALL = [1, 2, 3, 4, 5]
BAR = str(T.BAR_B2)


@pytest.mark.parametrize("folds", [[1], [2, 4], [1, 2, 3, 4]])
def test_partial_run_gets_no_yes_or_no(folds):
    line = L.verdict_line(0.844, folds, ALL, smoke=False)          # 0.844 is above the bar, and still no "Yes"
    assert line.startswith("**No verdict: partial run**")
    assert "Yes" not in line and "beats" not in line and "does not beat" not in line
    assert f"of {len(ALL)}" in line and ", ".join(map(str, folds)) in line
    assert "0.844" in line and BAR in line


def test_full_run_is_compared_with_the_all_fold_mean():
    yes = L.verdict_line(T.BAR_B2 + 0.01, ALL, ALL, smoke=False)
    no = L.verdict_line(T.BAR_B2 - 0.01, [5, 4, 3, 2, 1], ALL, smoke=False)   # the order of the folds does not matter
    assert yes.startswith("**Yes: beats the random forest**") and BAR in yes and "all 5 folds" in yes
    assert no.startswith("**No: does not beat the random forest yet**") and BAR in no
    assert not L.verdict_line(T.BAR_B2, ALL, ALL, smoke=False).startswith("**Yes")   # a tie is not a win


def test_smoke_run_keeps_the_smoke_warning():
    for folds in ([1], ALL):
        assert L.verdict_line(0.9, folds, ALL, smoke=True).startswith("SMOKE TEST")


def test_set_verdict_replaces_only_the_verdict_line():
    md = ("# DINOv3 (lora)\n\nintro\n\n"
          f"**Verdict:** **Yes: beats the random forest** (DINOv3 0.844 vs B2 {BAR})\n\n"
          "## Headline\n\n| Model | iou |\n")
    new = L.set_verdict(md, L.verdict_line(0.844, [1], ALL, smoke=False))
    assert "**Verdict:** **No verdict: partial run** (fold 1 of 5)" in new
    assert "Yes: beats" not in new and new.count("**Verdict:**") == 1
    assert new.split("**Verdict:**")[0] == md.split("**Verdict:**")[0]         # the text before it is untouched
    assert new.endswith("## Headline\n\n| Model | iou |\n")                    # and the text after it
