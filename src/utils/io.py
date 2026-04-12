"""
src/utils/io.py
---------------
Utility functions for reading and writing CSV / JSON result files.
"""

import csv
import json
import pathlib
from typing import Any


def write_csv(data: list[dict], output_path: str | pathlib.Path, fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts to a CSV file.

    Args:
        data: List of row dicts to write.
        output_path: Destination CSV path.
        fieldnames: Explicit column order; inferred from first row if None.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        output_path.write_text("")
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def read_csv(input_path: str | pathlib.Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts.

    Args:
        input_path: Path to the CSV file.

    Returns:
        List of row dicts (all values as strings).
    """
    with pathlib.Path(input_path).open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(data: Any, output_path: str | pathlib.Path, indent: int = 2) -> None:
    """Write a Python object to a JSON file.

    Args:
        data: JSON-serialisable object.
        output_path: Destination JSON path.
        indent: Pretty-print indent level.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(data, fh, indent=indent)


def read_json(input_path: str | pathlib.Path) -> Any:
    """Read a JSON file and return the parsed object.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Parsed Python object.
    """
    with pathlib.Path(input_path).open("r") as fh:
        return json.load(fh)


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Target directory path.

    Returns:
        Resolved pathlib.Path to the directory.
    """
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
