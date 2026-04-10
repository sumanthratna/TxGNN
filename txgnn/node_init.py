from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight tooling envs
    torch = None


EMBEDDING_VALUE_KEYS = ("embeddings", "matrix", "values", "features")
ID_KEYS = ("node_ids", "ids")
INDEX_KEYS = ("node_indices", "indices", "node_index")
EMBEDDING_COLUMN_PREFIXES = ("emb_", "dim_", "feature_", "value_")
NON_EMBEDDING_COLUMNS = {
    "node_type",
    "node_id",
    "node_index",
    "node_name",
    "matched_on",
    "source_file",
    "row_index",
}


def normalize_identifier(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def sanitize_node_type(node_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", node_type.lower()).strip("_")


def build_node_id_maps(df: pd.DataFrame) -> dict[str, dict[int, str]]:
    node_maps: dict[str, dict[int, str]] = {}
    for side in ("x", "y"):
        type_column = f"{side}_type"
        idx_column = f"{side}_idx"
        id_column = f"{side}_id"
        subset = df[[type_column, idx_column, id_column]].dropna().copy()
        subset[id_column] = subset[id_column].map(normalize_identifier)
        subset[idx_column] = subset[idx_column].astype(int)
        for node_type, group in subset.groupby(type_column, sort=False):
            mapping = node_maps.setdefault(node_type, {})
            for row in group.itertuples(index=False):
                mapping[int(getattr(row, idx_column))] = getattr(row, id_column)
    return {
        node_type: dict(sorted(mapping.items(), key=lambda item: item[0]))
        for node_type, mapping in node_maps.items()
    }


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_torch(path: Path) -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required to load .pt/.pth node init payloads."
        )
    return torch.load(path, map_location=torch.device("cpu"))


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _load_csv_spec(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    return _normalize_tabular_spec(df)


def _find_embedding_columns(df: pd.DataFrame) -> list[str]:
    prefixed = [
        column
        for column in df.columns
        if column.startswith(EMBEDDING_COLUMN_PREFIXES)
    ]
    if prefixed:
        return prefixed

    numeric_columns = [
        column
        for column in df.columns
        if column not in NON_EMBEDDING_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    return numeric_columns


def _normalize_tabular_spec(df: pd.DataFrame) -> dict[str, Any]:
    embedding_columns = _find_embedding_columns(df)
    if not embedding_columns:
        raise ValueError(
            "Could not identify embedding columns in the tabular node init payload."
        )

    spec: dict[str, Any] = {
        "embeddings": df[embedding_columns].to_numpy(dtype=np.float32),
    }
    if "node_id" in df.columns:
        spec["node_ids"] = [normalize_identifier(value) for value in df["node_id"]]
    if "node_index" in df.columns:
        spec["node_indices"] = [int(value) for value in df["node_index"]]
    if "node_type" in df.columns and df["node_type"].nunique() == 1:
        spec["node_type"] = str(df["node_type"].iloc[0])
    return spec


def _normalize_single_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, pd.DataFrame):
        return _normalize_tabular_spec(spec)

    if torch is not None and isinstance(spec, torch.Tensor):
        return {"embeddings": spec.detach().cpu().numpy().astype(np.float32)}

    if isinstance(spec, np.ndarray):
        return {"embeddings": spec.astype(np.float32)}

    if isinstance(spec, list):
        return {"embeddings": np.asarray(spec, dtype=np.float32)}

    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported node init payload type: {type(spec)!r}")

    for nested_key in ("node_init", "node_embeddings", "payload"):
        if nested_key in spec and isinstance(spec[nested_key], dict):
            spec = spec[nested_key]
            break

    if "dataframe" in spec and isinstance(spec["dataframe"], pd.DataFrame):
        return _normalize_tabular_spec(spec["dataframe"])

    normalized: dict[str, Any] = {}
    for value_key in EMBEDDING_VALUE_KEYS:
        if value_key in spec:
            value = spec[value_key]
            if torch is not None and isinstance(value, torch.Tensor):
                normalized["embeddings"] = value.detach().cpu().numpy().astype(
                    np.float32
                )
            else:
                normalized["embeddings"] = np.asarray(value, dtype=np.float32)
            break

    if "embeddings" not in normalized and {
        "node_index",
    }.issubset(spec.keys()):
        return _normalize_tabular_spec(pd.DataFrame(spec))

    for key in ID_KEYS:
        if key in spec:
            normalized["node_ids"] = [
                normalize_identifier(value) for value in spec[key]
            ]
            break

    for key in INDEX_KEYS:
        if key in spec:
            normalized["node_indices"] = [int(value) for value in spec[key]]
            break

    if "node_type" in spec:
        normalized["node_type"] = str(spec["node_type"])

    if "embeddings" not in normalized:
        raise ValueError(
            "Node init payload is missing an embedding matrix under one of: "
            + ", ".join(EMBEDDING_VALUE_KEYS)
        )
    return normalized


def _normalize_payload_by_type(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_spec in payload.items():
        if raw_key in ("node_init", "node_embeddings", "payload"):
            nested = payload[raw_key]
            if isinstance(nested, dict):
                return _normalize_payload_by_type(nested)
        node_type = str(raw_key)
        normalized[node_type] = _normalize_single_spec(raw_spec)
    return normalized


def load_node_init_payload(path_like: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Node init payload {path} does not exist.")

    if path.is_dir():
        manifest_path = path / "manifest.json"
        payload_by_type: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            entries = manifest.get("node_types", {})
            for node_type, raw_spec in entries.items():
                file_name = raw_spec["file"] if isinstance(raw_spec, dict) else raw_spec
                payload_by_type[node_type] = _load_csv_spec(path / file_name)
        else:
            for csv_path in sorted(path.glob("*.csv")):
                payload_by_type[csv_path.stem] = _load_csv_spec(csv_path)
        return _normalize_payload_by_type(payload_by_type)

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        payload = _load_pickle(path)
    elif suffix in {".pt", ".pth"}:
        payload = _load_torch(path)
    elif suffix == ".npz":
        payload = _load_npz(path)
    elif suffix == ".csv":
        tabular = _load_csv_spec(path)
        node_type = tabular.get("node_type", "default")
        return {node_type: tabular}
    else:
        raise ValueError(
            f"Unsupported node init payload format {suffix!r} for {path}."
        )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a dictionary-like node init payload in {path}, got "
            f"{type(payload)!r}."
        )
    if any(key in payload for key in EMBEDDING_VALUE_KEYS):
        spec = _normalize_single_spec(payload)
        node_type = spec.get("node_type", "default")
        return {node_type: spec}
    return _normalize_payload_by_type(payload)


def infer_node_init_width(path_like: str | Path) -> int | None:
    payload = load_node_init_payload(path_like)
    widths = {
        int(spec["embeddings"].shape[1])
        for spec in payload.values()
        if len(spec["embeddings"].shape) == 2
    }
    if len(widths) == 1:
        return widths.pop()
    return None


def _resolve_spec_for_type(
    payload: dict[str, dict[str, Any]], node_type: str
) -> dict[str, Any] | None:
    if node_type in payload:
        return payload[node_type]
    sanitized = sanitize_node_type(node_type)
    if sanitized in payload:
        return payload[sanitized]
    for raw_key, spec in payload.items():
        if sanitize_node_type(raw_key) == sanitized:
            return spec
    return None


def resolve_node_init_tensors(
    node_init_path: str | Path,
    node_id_maps: dict[str, dict[int, str]],
    num_nodes_by_type: dict[str, int],
    n_inp: int,
    strict: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int | str]]]:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required to resolve node init tensors for TxGNN."
        )
    payload = load_node_init_payload(node_init_path)
    embeddings_by_type: dict[str, torch.Tensor] = {}
    summary: dict[str, dict[str, int | str]] = {}

    for node_type, num_nodes in num_nodes_by_type.items():
        spec = _resolve_spec_for_type(payload, node_type)
        if spec is None:
            if strict:
                raise KeyError(
                    f"Node init payload {node_init_path} is missing a spec for "
                    f"node type {node_type!r}."
                )
            continue

        matrix = np.asarray(spec["embeddings"], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"Node init embeddings for {node_type!r} must be 2-D, got shape "
                f"{matrix.shape}."
            )
        if matrix.shape[1] != n_inp:
            raise ValueError(
                f"Node init embeddings for {node_type!r} have width {matrix.shape[1]}, "
                f"but TxGNN was initialized with n_inp={n_inp}."
            )

        tensor = torch.empty((num_nodes, n_inp), dtype=torch.float32)
        nn_init = torch.nn.init.xavier_uniform_
        nn_init(tensor)

        matched_rows = 0
        unmatched_rows = num_nodes
        matched_on = "row_order"

        if "node_indices" in spec:
            matched_on = "node_index"
            for row_index, node_index in enumerate(spec["node_indices"]):
                if 0 <= int(node_index) < num_nodes:
                    tensor[int(node_index)] = torch.from_numpy(matrix[row_index])
                    matched_rows += 1
        elif "node_ids" in spec:
            matched_on = "node_id"
            id_to_index = {
                normalize_identifier(node_id): int(index)
                for index, node_id in node_id_maps.get(node_type, {}).items()
            }
            for row_index, node_id in enumerate(spec["node_ids"]):
                mapped_index = id_to_index.get(normalize_identifier(node_id))
                if mapped_index is None:
                    continue
                tensor[mapped_index] = torch.from_numpy(matrix[row_index])
                matched_rows += 1
        else:
            if matrix.shape[0] != num_nodes:
                raise ValueError(
                    f"Row-order node init for {node_type!r} has {matrix.shape[0]} rows "
                    f"but the graph expects {num_nodes}."
                )
            tensor[:] = torch.from_numpy(matrix)
            matched_rows = num_nodes

        unmatched_rows = num_nodes - matched_rows
        if strict and unmatched_rows > 0:
            raise ValueError(
                f"Strict node init requested for {node_type!r}, but only "
                f"{matched_rows}/{num_nodes} rows were matched."
            )

        embeddings_by_type[node_type] = tensor
        summary[node_type] = {
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
            "num_nodes": num_nodes,
            "embedding_dim": n_inp,
            "matched_on": matched_on,
        }

    return embeddings_by_type, summary
