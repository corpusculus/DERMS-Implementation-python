"""
src/analysis/run_dashboard.py
-------------------------------
CLI entry point for generating interactive Plotly dashboards from QSTS results.

Usage:
    python -m src.analysis.run_dashboard --results-dir results
"""

import argparse
import pathlib
import sys

# Ensure project root is importable
_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.dashboard import create_dashboard


def main() -> None:
    """Generate the interactive dashboard."""
    args = _parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  DERMS Dashboard Generator")
    print(f"{'='*60}")
    print(f"  Results dir: {results_dir}")

    try:
        output_path = pathlib.Path(create_dashboard(
            results_dir=results_dir,
            output_path=pathlib.Path(args.output) if args.output else None,
        ))

        print(f"  Output: {output_path}")
        print(f"{'='*60}\n")
        print(f"✓ Dashboard generated successfully!")
        print(f"  Open in browser: file://{output_path.resolve()}")

    except Exception as e:
        print(f"\n✗ Error generating dashboard: {e}")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive Plotly dashboard from QSTS results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing QSTS result CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: <results-dir>/dashboard.html).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
