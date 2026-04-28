"""
Mirrors train_camargo_LSTM.ipynb so we can run training from the shell
(parameterized for a quick smoke test vs. the full 100-epoch run).
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
# project root = .../XAI-Probabilistic_Suffix_Prediction_U-ED-LSTM_pub
PROJECT_ROOT = HERE.parents[5]
# package root = .../camargo_LSTM_suffix_pred (so `joinLSTM` and `training` import)
PKG_ROOT = HERE.parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PKG_ROOT))

from joinLSTM.model import FullShared_Join_LSTM
from training.train import Training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--saving-path", type=str, required=True)
    parser.add_argument("--writer-comment", type=str, required=True)
    args = parser.parse_args()

    train_pkl = PROJECT_ROOT / "encoded_data/compare_camargo/helpdesk_all_5_train.pkl"
    val_pkl = PROJECT_ROOT / "encoded_data/compare_camargo/helpdesk_all_5_val.pkl"

    train_ds = torch.load(train_pkl, weights_only=False)
    val_ds = torch.load(val_pkl, weights_only=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    all_categories = train_ds.all_categories
    cats_meta, nums_meta = all_categories

    concept_name = "Activity"
    concept_name_id = next(i for i, c in enumerate(cats_meta) if c[0] == concept_name)
    concept_name_size = cats_meta[concept_name_id][1]
    eos_id = cats_meta[concept_name_id][2]["EOS"]

    model_feat = [[c[0] for c in cats_meta], [n[0] for n in nums_meta]]

    model = FullShared_Join_LSTM(
        data_set_categories=all_categories,
        hidden_size=50,
        num_layers=1,
        model_feat=model_feat,
        input_size=1,
        output_size_act=concept_name_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4, min_lr=1e-10)

    writer = SummaryWriter(comment=args.writer_comment)

    optimize_values = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epochs": args.epochs,
        "mini_batches": 128,
        "shuffle": True,
    }

    saving_path = str(HERE / args.saving_path)

    trainer = Training(
        model=model,
        device=device,
        data_train=train_ds,
        data_val=val_ds,
        optimize_values=optimize_values,
        concept_name_id=concept_name_id,
        eos_id=eos_id,
        writer=writer,
        save_model_n_th_epoch=1,
        saving_path=saving_path,
    )

    trainer.train()


if __name__ == "__main__":
    main()
