"""
Score infer/label pairs in a CSV: word accuracy and jaso-level 1-NED.

A wildcard in the label matches zero or one predicted token at no cost,
so illegible characters do not count against the model.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from kstrbench.step import Step

DEFAULT_CSV_ENCODING = "utf-8-sig"
DEFAULT_WILDCARD = "□"

CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSUNG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3


def to_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def normalize_text(text: str, ignore_blank: bool, is_lower: bool) -> str:
    out = text.strip()
    if ignore_blank:
        out = "".join(out.split())
    return out.lower() if is_lower else out


def text_to_jaso(text: str, wildcard: str) -> List[str]:
    jaso: List[str] = []
    for char in text:
        if char == wildcard:
            jaso.extend([wildcard] * 3)
        elif HANGUL_BASE <= ord(char) <= HANGUL_LAST:
            code = ord(char) - HANGUL_BASE
            jaso.append(CHOSUNG[code // 588])
            jaso.append(JUNGSUNG[(code % 588) // 28])
            if code % 28:
                jaso.append(JONGSUNG[code % 28 - 1])
        else:
            jaso.append(char)
    return jaso


def edit_distance(gt: Sequence[Any], pred: Sequence[Any]) -> int:
    n, m = len(gt), len(pred)
    if n == 0 or m == 0:
        return max(n, m)

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        gt_token = gt[i - 1]
        for j in range(1, m + 1):
            cost = 0 if gt_token == pred[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return int(prev[m])


def wildcard_edit_distance(
    gt: Sequence[Any],
    pred: Sequence[Any],
    wildcard: Any,
) -> int:
    """Each wildcard in gt absorbs zero or one predicted token at no cost."""
    n, m = len(gt), len(pred)
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == inf:
                continue
            if j < m:
                dp[i][j + 1] = min(dp[i][j + 1], dp[i][j] + 1)
            if i == n:
                continue
            if gt[i] == wildcard:
                dp[i + 1][j] = min(dp[i + 1][j], dp[i][j])
                if j < m:
                    dp[i + 1][j + 1] = min(dp[i + 1][j + 1], dp[i][j])
            else:
                dp[i + 1][j] = min(dp[i + 1][j], dp[i][j] + 1)
                if j < m:
                    cost = 0 if gt[i] == pred[j] else 1
                    dp[i + 1][j + 1] = min(dp[i + 1][j + 1], dp[i][j] + cost)
    return int(dp[n][m])


def one_minus_ned(gt: Sequence[Any], pred: Sequence[Any], wildcard: str | None) -> float:
    if wildcard is None:
        dist = edit_distance(gt, pred)
        denom = len(gt)
    else:
        dist = wildcard_edit_distance(gt, pred, wildcard)
        denom = sum(1 for token in gt if token != wildcard)
    return 1.0 - dist / max(denom, 1, len(pred))


def compute_metrics(
    label_text: str,
    infer_text: str,
    ignore_blank: bool,
    is_lower: bool,
    wildcard: str,
) -> tuple[int, float]:
    label = normalize_text(label_text, ignore_blank, is_lower)
    infer = normalize_text(infer_text, ignore_blank, is_lower)
    if label == infer:
        return 1, 1.0

    use_wildcard = wildcard in label or wildcard in infer
    active_wildcard = wildcard if use_wildcard else None
    if use_wildcard:
        is_match = 1 if wildcard_edit_distance(label, infer, wildcard) == 0 else 0
    else:
        is_match = 0

    label_jaso = text_to_jaso(label, wildcard)
    infer_jaso = text_to_jaso(infer, wildcard)
    if label_jaso == infer_jaso:
        return is_match, 1.0
    return is_match, one_minus_ned(label_jaso, infer_jaso, active_wildcard)


def execute(
    csv_path: Path,
    output_csv_path: Path | None = None,
    infer_col: str = "infer",
    label_col: str = "label",
    match_col: str = "is_match",
    jaso_score_col: str = "jaso_1_ned",
    ignore_blank: bool = True,
    is_lower: bool = False,
    wildcard: str = DEFAULT_WILDCARD,
    csv_encoding: str = DEFAULT_CSV_ENCODING,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_csv = output_csv_path or csv_path

    with csv_path.open("r", encoding=csv_encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = {infer_col, label_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing columns {sorted(missing)}: {csv_path}. "
                f"Available: {reader.fieldnames}"
            )
        fieldnames = list(reader.fieldnames)
        for col in (match_col, jaso_score_col):
            if col not in fieldnames:
                fieldnames.append(col)
        rows: List[Dict[str, Any]] = list(reader)

    match_sum = 0
    jaso_sum = 0.0
    for row in rows:
        is_match, jaso_score = compute_metrics(
            label_text=to_text(row.get(label_col, "")),
            infer_text=to_text(row.get(infer_col, "")),
            ignore_blank=ignore_blank,
            is_lower=is_lower,
            wildcard=wildcard,
        )
        row[match_col] = is_match
        row[jaso_score_col] = jaso_score
        match_sum += is_match
        jaso_sum += jaso_score

    denom = max(len(rows), 1)
    accuracy = match_sum / denom
    jaso_1_ned = jaso_sum / denom

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding=csv_encoding, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"csv_path        : {csv_path}")
    print(f"output_csv_path : {out_csv}")
    print(f"label_col       : {label_col}")
    print(f"infer_col       : {infer_col}")
    print(f"ignore_blank    : {ignore_blank}")
    print(f"is_lower        : {is_lower}")
    print(f"wildcard        : {wildcard}")
    print(f"rows            : {len(rows)}")
    print(f"accuracy        : {accuracy:.6f}")
    print(f"jaso_1_ned      : {jaso_1_ned:.6f}")

    return {
        "csv_path": str(csv_path),
        "output_csv_path": str(out_csv),
        "infer_col": infer_col,
        "label_col": label_col,
        "match_col": match_col,
        "jaso_score_col": jaso_score_col,
        "ignore_blank": ignore_blank,
        "is_lower": is_lower,
        "wildcard": wildcard,
        "csv_encoding": csv_encoding,
        "rows": len(rows),
        "accuracy": accuracy,
        "jaso_1_ned": jaso_1_ned,
    }


@dataclass(kw_only=True)
class ScoreInferCsvStep(Step):
    csv_path: str
    output_csv_path: str | None = None
    infer_col: str = "infer"
    label_col: str = "label"
    match_col: str = "is_match"
    jaso_score_col: str = "jaso_1_ned"
    ignore_blank: bool = True
    is_lower: bool = False
    wildcard: str = DEFAULT_WILDCARD
    csv_encoding: str = DEFAULT_CSV_ENCODING

    def run(self) -> Dict[str, Any]:
        return execute(
            csv_path=Path(self.csv_path),
            output_csv_path=Path(self.output_csv_path) if self.output_csv_path else None,
            infer_col=self.infer_col,
            label_col=self.label_col,
            match_col=self.match_col,
            jaso_score_col=self.jaso_score_col,
            ignore_blank=self.ignore_blank,
            is_lower=self.is_lower,
            wildcard=self.wildcard,
            csv_encoding=self.csv_encoding,
        )
