#!/usr/bin/env python
"""Run full_enc_dec_lstm_gn.ipynb with CLI-configurable train/val pickle paths.

The training notebook hard-codes `file_path_train` / `file_path_val`. This wrapper
loads the notebook in memory, rewrites that single cell to point at the paths
passed on the command line, and executes the notebook end-to-end via nbclient
(same kernel, same outputs, same side effects — TensorBoard logs and the
checkpoint pickle still land where the notebook puts them).

The training notebook does not load a test pickle; only train + val are used
during training. Test pickles are consumed by the evaluation notebooks.

Examples
--------
    pipenv run python run_training.py \
        --train ../../BPIC_2017_all_5_train_augmented_A1_A2_A3.pkl \
        --val   ../../Loader/pkl/BPIC_2017_all_5_val.pkl

    # masking experiment: train + val both masked
    pipenv run python run_training.py \
        --train ../../BPIC_2017_all_5_train_augmented_A1_A2_A3.pkl \
        --val   ../../BPIC_2017_all_5_val_M.pkl \
        --save-path ../pkl/BPIC_2017_full_grad_norm_A1_A2_A3_model.pkl \
        

    pipenv run python run_training.py \
    --train ../../BPIC_2017_all_5_train_augmented_M.pkl \
    --val   ../../BPIC_2017_all_5_val_M.pkl \
    --save-path ../pkl/BPIC_2017_full_grad_norm_M_model.pkl \
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK_PATH = Path(__file__).resolve().parent / "full_enc_dec_lstm_gn.ipynb"


def build_data_cell(train_path: Path, val_path: Path) -> str:
    """Return source for the cell that loads train + val datasets.

    Replaces the notebook's USE_AUGMENTED_TRAIN branching so the runner has
    sole control of which pickles are read. Absolute paths are used so the
    cell works regardless of the notebook's cwd.
    """
    return (
        "import sys\n"
        "sys.path.insert(0, '../../../../../../..')  # -> src/\n"
        "import torch\n"
        "\n"
        f"file_path_train = {str(train_path)!r}\n"
        f"file_path_val   = {str(val_path)!r}\n"
        "print(f'Train pickle: {file_path_train}')\n"
        "print(f'Val pickle:   {file_path_val}')\n"
        "\n"
        "BPIC_17_train_dataset = torch.load(file_path_train, weights_only=False)\n"
        "print(type(BPIC_17_train_dataset))\n"
        "BPIC_17_val_dataset = torch.load(file_path_val, weights_only=False)\n"
        "print(type(BPIC_17_val_dataset))\n"
    )


def patch_data_cell(nb, train_path: Path, val_path: Path) -> None:
    """Find the data-loading cell and replace its source. Raises if not found."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if "BPIC_17_train_dataset" in src and "torch.load" in src:
            cell.source = build_data_cell(train_path, val_path)
            return
    raise RuntimeError(
        "Could not find the data-loading cell (expected one referencing "
        "`BPIC_17_train_dataset` and `torch.load`)."
    )


def patch_trainer_cell(nb, save_path: Path | None, tb_comment: str | None) -> None:
    """Optionally rewrite saving_path and SummaryWriter comment in the trainer cell."""
    if save_path is None and tb_comment is None:
        return
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if "saving_path" in src and "Trainer(" in src:
            if save_path is not None:
                src = re.sub(
                    r"saving_path\s*=\s*['\"][^'\"]*['\"]",
                    f"saving_path = {str(save_path)!r}",
                    src,
                    count=1,
                )
            if tb_comment is not None:
                src = re.sub(
                    r'comment\s*=\s*"[^"]*"',
                    f'comment="{tb_comment}"',
                    src,
                    count=1,
                )
            cell.source = src
            return
    raise RuntimeError(
        "Could not find the Trainer cell (expected one referencing "
        "`saving_path` and `Trainer(`)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the BPIC17 training notebook with custom data paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", required=True, type=Path, help="Train pickle path")
    parser.add_argument("--val", required=True, type=Path, help="Val pickle path")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Override checkpoint output path (default: keep notebook's "
             "'../pkl/BPIC_2017_full_grad_norm_philipp_final_run.pkl')",
    )
    parser.add_argument(
        "--tb-comment",
        type=str,
        default=None,
        help="Override SummaryWriter `comment=` (TensorBoard run-folder suffix). "
             "Default: keep notebook's 'Full_BPIC17_grad'.",
    )
    parser.add_argument(
        "--executed-out",
        type=Path,
        default=None,
        help="If set, write the executed notebook (with outputs) to this path.",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=None,
        help="Per-cell timeout in seconds (default: no timeout — training cells take hours).",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="python3",
        help="Jupyter kernel name (default: python3).",
    )
    args = parser.parse_args()

    train_path = args.train.resolve()
    val_path = args.val.resolve()
    if not train_path.is_file():
        print(f"error: train pickle not found: {train_path}", file=sys.stderr)
        return 2
    if not val_path.is_file():
        print(f"error: val pickle not found: {val_path}", file=sys.stderr)
        return 2

    save_path = args.save_path.resolve() if args.save_path else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Notebook:    {NOTEBOOK_PATH}")
    print(f"Train pkl:   {train_path}")
    print(f"Val pkl:     {val_path}")
    if save_path:
        print(f"Save path:   {save_path}")
    if args.tb_comment:
        print(f"TB comment:  {args.tb_comment}")
    print()

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    patch_data_cell(nb, train_path, val_path)
    patch_trainer_cell(nb, save_path, args.tb_comment)

    client = NotebookClient(
        nb,
        timeout=args.cell_timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        print("\n--- Notebook execution failed ---", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        if args.executed_out:
            nbformat.write(nb, str(args.executed_out))
            print(f"\nPartial executed notebook written to {args.executed_out}",
                  file=sys.stderr)
        return 1

    if args.executed_out:
        out = args.executed_out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, str(out))
        print(f"\nExecuted notebook written to {out}")

    print("\nTraining finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
