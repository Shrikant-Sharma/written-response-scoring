"""Compare the headline numbers of your run with the run this folder was shipped with.

    python check_my_run.py

Reads results/summary.json, which 05_report.py writes at the end of a run, and reference_run/summary.json, which
is never overwritten. Prints how many numbers match and lists every one that differs. A count that differs by a
few can come from another processor or another library version. The numbers to quote are the ones your run shows.
"""
import sys

from common import RESULTS, ROOT, read_json


def flat(node, path=""):
    """Every number in a nested dictionary, keyed by its path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flat(value, f"{path} / {key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from flat(value, f"{path} [{i}]")
    elif isinstance(node, bool) or node is None or isinstance(node, str):
        yield path, node
    else:
        yield path, float(node)


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def main():
    mine_path, reference_path = RESULTS / "summary.json", ROOT / "reference_run" / "summary.json"
    if not mine_path.exists():
        sys.exit("results/summary.json is missing. Run the steps through 05_report.py first.")
    if not reference_path.exists():
        sys.exit("reference_run/summary.json is missing, so there is nothing to compare with.")
    mine, reference = dict(flat(read_json(mine_path))), dict(flat(read_json(reference_path)))
    keys = [k for k in reference if k in mine]
    different = [k for k in keys if not same(mine[k], reference[k])]
    missing = [k for k in reference if k not in mine] + [k for k in mine if k not in reference]
    print(f"{len(keys) - len(different)} of {len(keys)} numbers are the same as in the reference run.")
    if different:
        print(f"\n{len(different)} differ:\n")
        width = max(len(k) for k in different)
        print(f"{'number'.ljust(width)}   {'your run':>14}   {'reference':>14}")
        for k in different:
            a, b = mine[k], reference[k]
            show = (lambda v: f"{v:,.4f}" if isinstance(v, float) and v != int(v) else f"{int(v):,}" if isinstance(v, float) else str(v))
            print(f"{k.ljust(width)}   {show(a):>14}   {show(b):>14}")
        print("\nQuote the numbers of your own run.")
    if missing:
        print(f"\n{len(missing)} entries exist in only one of the two files:")
        for k in missing:
            print("  " + k)


if __name__ == "__main__":
    main()
