from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..helpers.param_utils import coerce_bool
from ..helpers.step_logger import StepLogger, emit, normalize_log_level


class RunDataFrameOperationsStep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    def run(self) -> dict[str, Any]:
        return run_from_config(self.params)


def run() -> None:
    run_from_config({})


def _as_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _allowed_eval_env() -> dict[str, Any]:
    allowed: dict[str, Any] = {"abs": abs, "min": min, "max": max, "round": round, "pow": pow}
    for name in (
        "log",
        "log2",
        "log10",
        "sqrt",
        "exp",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "floor",
        "ceil",
        "fabs",
    ):
        allowed[name] = getattr(math, name)
    allowed["pi"] = math.pi
    allowed["e"] = math.e
    return allowed


def _to_bool_list(raw: Any, expected_len: int) -> list[bool]:
    if isinstance(raw, (list, tuple)):
        out = [coerce_bool(v, default=True) for v in raw]
        if len(out) != expected_len:
            raise ValueError(f"ascending length mismatch: expected {expected_len}, got {len(out)}")
        return out
    one = coerce_bool(raw, default=True)
    return [one for _ in range(expected_len)]


def _apply_filter(df: Any, op: dict[str, Any]) -> Any:
    column = str(op.get("column", "")).strip()
    operator = str(op.get("operator", op.get("op", "eq"))).strip().lower()
    value = op.get("value")
    values = _as_list(op.get("values"))
    if not column:
        raise ValueError("filter operation requires 'column'")
    if column not in df.columns:
        raise ValueError(f"filter column not found: {column}")

    series = df[column]
    if operator == "eq":
        mask = series.astype(str) == str(value)
    elif operator == "ne":
        mask = series.astype(str) != str(value)
    elif operator == "in":
        if not values:
            raise ValueError("filter in requires 'values'")
        mask = series.astype(str).isin([str(v) for v in values])
    elif operator == "not_in":
        if not values:
            raise ValueError("filter not_in requires 'values'")
        mask = ~series.astype(str).isin([str(v) for v in values])
    elif operator in ("gt", "gte", "lt", "lte"):
        num = series.astype(float)
        rhs = float(value)
        if operator == "gt":
            mask = num > rhs
        elif operator == "gte":
            mask = num >= rhs
        elif operator == "lt":
            mask = num < rhs
        else:
            mask = num <= rhs
    elif operator == "contains":
        mask = series.astype(str).str.contains(str(value), regex=False, na=False)
    elif operator == "not_contains":
        mask = ~series.astype(str).str.contains(str(value), regex=False, na=False)
    else:
        raise ValueError(f"unsupported filter operator: {operator}")
    return df[mask]


def run_from_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    in_csv_file_raw = params.get("in_csv_file")
    out_csv_file_raw = params.get("out_csv_file")
    operations = _as_list(params.get("operations"))
    save_index = coerce_bool(params.get("save_index", False), default=False)
    log_level = normalize_log_level(params.get("log_level"), default="INFO")

    if not in_csv_file_raw:
        raise ValueError("in_csv_file is required")
    if not out_csv_file_raw:
        raise ValueError("out_csv_file is required")
    if not operations:
        raise ValueError("operations is required")

    in_csv_file = Path(str(in_csv_file_raw)).expanduser().resolve()
    out_csv_file = Path(str(out_csv_file_raw)).expanduser().resolve()
    if not in_csv_file.is_file():
        raise FileNotFoundError(f"in_csv_file not found: {in_csv_file}")

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for run_dataframe_operations_entry") from exc

    logger = StepLogger("run_dataframe_operations_entry")
    logger.log(
        {
            "in_csv_file": str(in_csv_file),
            "out_csv_file": str(out_csv_file),
            "operation_count": len(operations),
            "operations": operations,
            "save_index": save_index,
            "log_level": log_level,
        },
        title="dataframe operations discovery",
    )

    df = pd.read_csv(in_csv_file, encoding="utf-8-sig")
    row_count_before = int(len(df))
    env = _allowed_eval_env()

    for i, raw_op in enumerate(operations):
        if not isinstance(raw_op, dict):
            raise ValueError(f"operations[{i}] must be an object")
        op_type = str(raw_op.get("type", raw_op.get("op", ""))).strip().lower()
        if not op_type:
            raise ValueError(f"operations[{i}] requires 'type'")

        if op_type == "filter":
            df = _apply_filter(df, raw_op)
        elif op_type == "select_columns":
            cols = [str(c) for c in _as_list(raw_op.get("columns"))]
            if not cols:
                raise ValueError(f"operations[{i}] select_columns requires 'columns'")
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"operations[{i}] missing columns: {missing}")
            df = df[cols]
        elif op_type == "rename_columns":
            mapping = raw_op.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"operations[{i}] rename_columns requires non-empty 'mapping'")
            df = df.rename(columns={str(k): str(v) for k, v in mapping.items()})
        elif op_type == "sort_values":
            by = [str(c) for c in _as_list(raw_op.get("by"))]
            if not by:
                raise ValueError(f"operations[{i}] sort_values requires 'by'")
            missing = [c for c in by if c not in df.columns]
            if missing:
                raise ValueError(f"operations[{i}] sort_values missing columns: {missing}")
            asc = _to_bool_list(raw_op.get("ascending", True), expected_len=len(by))
            df = df.sort_values(by=by, ascending=asc)
        elif op_type == "drop_duplicates":
            subset = [str(c) for c in _as_list(raw_op.get("subset"))]
            keep = str(raw_op.get("keep", "first")).strip().lower()
            if subset:
                missing = [c for c in subset if c not in df.columns]
                if missing:
                    raise ValueError(f"operations[{i}] drop_duplicates missing columns: {missing}")
                df = df.drop_duplicates(subset=subset, keep=keep)
            else:
                df = df.drop_duplicates(keep=keep)
        elif op_type == "assign_constant":
            column = str(raw_op.get("column", "")).strip()
            if not column:
                raise ValueError(f"operations[{i}] assign_constant requires 'column'")
            df[column] = raw_op.get("value")
        elif op_type == "assign_formula":
            out_column = str(raw_op.get("out_column", raw_op.get("column", ""))).strip()
            formula = str(raw_op.get("formula", "")).strip()
            source_column = str(raw_op.get("source_column", "")).strip()
            precision_raw = raw_op.get("precision")
            if not out_column:
                raise ValueError(f"operations[{i}] assign_formula requires 'out_column'")
            if not formula:
                raise ValueError(f"operations[{i}] assign_formula requires 'formula'")
            formula_code = compile(formula, "<formula>", "eval")
            if source_column:
                if source_column not in df.columns:
                    raise ValueError(f"operations[{i}] source_column not found: {source_column}")
                source_series = df[source_column]
            else:
                source_series = None

            values: list[Any] = []
            for _, row in df.iterrows():
                local_env = dict(env)
                if source_series is not None:
                    x = float(row[source_column])
                    local_env["x"] = x
                for c in df.columns:
                    local_env[str(c)] = row[c]
                out_val = eval(formula_code, {"__builtins__": {}}, local_env)
                values.append(out_val)
            if precision_raw is not None:
                precision = max(0, int(precision_raw))
                rounded: list[Any] = []
                for v in values:
                    if isinstance(v, (int, float)):
                        rounded.append(f"{float(v):.{precision}f}")
                    else:
                        rounded.append(v)
                df[out_column] = rounded
            else:
                df[out_column] = values
        else:
            raise ValueError(f"unsupported operation type: {op_type}")

    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_file, index=save_index, encoding="utf-8-sig")
    summary = {
        "ok": True,
        "in_csv_file": str(in_csv_file),
        "out_csv_file": str(out_csv_file),
        "operation_count": len(operations),
        "row_count_before": row_count_before,
        "row_count_after": int(len(df)),
        "columns_after": [str(c) for c in df.columns],
        "save_index": save_index,
        "log_level": log_level,
    }
    logger.log(summary, title="dataframe operations summary")
    emit(log_level, "INFO", f"[df-ops] rows {row_count_before} -> {len(df)}")
    emit(log_level, "INFO", f"[df-ops] out_csv={out_csv_file}")
    return summary

