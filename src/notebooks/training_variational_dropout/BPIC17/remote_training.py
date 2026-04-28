"""
Headless remote-server mirror of
src/notebooks/training_variational_dropout/BPIC17/full_enc_dec_lstm_gn.ipynb

Differences vs. the notebook:
- No plotting window (final loss curve saved to PNG).
- stdout + stderr duplicated to a timestamped log file next to this script.
- Per-epoch checkpoint pkls land as <saving_path>_epoch<N>.pkl via the Trainer.
- A separate tab-separated file with just per-epoch train/val losses is written
  at the end for easy parsing / plotting later.

Run from anywhere; paths are resolved relative to this file.
"""

import os
import sys
import datetime


# Flip to True for a ~1-min sanity run (tiny subset, shrunk model, 1 epoch)
# before pushing to the server. Leave False for real training runs.
SMOKE_TEST = True


HERE = os.path.dirname(os.path.abspath(__file__))
# BPIC17/ is 3 levels below src/ — add src/ to sys.path so model/, loss/,
# trainer/, event_log_loader/ are importable (the last one is required
# before torch.load, because the pickle references EventLogDataset).
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        # Delegate to the primary (terminal) stream so IPython/tqdm see
        # the underlying tty status.
        return self.streams[0].isatty() if self.streams else False

    def __getattr__(self, name):
        # Any other attribute probes (encoding, fileno, etc.) fall through
        # to the primary stream.
        return getattr(self.streams[0], name)


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(HERE, f"remote_training_log_{timestamp}.txt")
_log_file = open(log_path, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

print(f"# remote_training.py started at {timestamp}")
print(f"# log file: {log_path}")


import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


file_path_train = os.path.join(HERE, '..', '..', '..', '..', 'encoded_data', 'BPIC_2017_all_5_train.pkl')
file_path_val = os.path.join(HERE, '..', '..', '..', '..', 'encoded_data', 'BPIC_2017_all_5_val.pkl')

BPIC_17_train_dataset = torch.load(file_path_train, weights_only=False)
print(type(BPIC_17_train_dataset))
BPIC_17_val_dataset = torch.load(file_path_val, weights_only=False)
print(type(BPIC_17_val_dataset))

bpic_17_all_categories = BPIC_17_train_dataset.all_categories
bpic_17_all_categories_cat = bpic_17_all_categories[0]
bpic_17_all_categories_num = bpic_17_all_categories[1]
print(bpic_17_all_categories_cat)
print(bpic_17_all_categories_num)

for i, cat in enumerate(bpic_17_all_categories_cat):
    print(f"BPIC 17 (5) Categorical feature: {cat[0]}, Index position in categorical data list: {i}")
    print(f"BPIC 17 (5) Total Amount of Category labels: {cat[1]}")
print('\n')
for i, num in enumerate(bpic_17_all_categories_num):
    print(f"BPIC 17 (5) Numerical feature: {num[0]}, Index position in categorical data list: {i}")
    print(f"BPIC 17 (5) Amount Category Lables: {num[1]}")

enc_feat_cat = [cat[0] for cat in bpic_17_all_categories_cat]
enc_feat_num = [num[0] for num in bpic_17_all_categories_num]
enc_feat = [enc_feat_cat, enc_feat_num]
print("Input features encoder: ", enc_feat)

dec_feat_cat = ['concept:name', 'org:resource', 'lifecycle:transition']
dec_feat_num = ['case_elapsed_time', 'event_elapsed_time']
dec_feat = [dec_feat_cat, dec_feat_num]
print("Features decoder: ", dec_feat)


if SMOKE_TEST:
    # Dataset attributes (`.all_categories`) were already pulled above —
    # safe to wrap in Subset, which hides them.
    from torch.utils.data import Subset
    BPIC_17_train_dataset = Subset(BPIC_17_train_dataset, range(500))
    BPIC_17_val_dataset = Subset(BPIC_17_val_dataset, range(200))
    print("# SMOKE_TEST on — data subsampled to 500/200")


from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import DropoutUncertaintyEncoderDecoderLSTM

seq_len_pred = 4
hidden_size = 32 if SMOKE_TEST else 128
num_layers = 1 if SMOKE_TEST else 4
dropout = 0.0

model = DropoutUncertaintyEncoderDecoderLSTM(
    data_set_categories=bpic_17_all_categories,
    enc_feat=enc_feat,
    dec_feat=dec_feat,
    seq_len_pred=seq_len_pred,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
)


from loss.losses import Loss
loss_obj = Loss()


from trainer.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment="Full_BPIC17_grad_remote")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

learning_rate = 1e-6
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-10)

num_epochs = 1 if SMOKE_TEST else 100
batch_size = 256
regularization_term = 1e-4
shuffle = True
teacher_forcing_ratio = 0.8

optimize_values = {
    "regularization_term": regularization_term,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "epochs": num_epochs,
    "mini_batches": batch_size,
    "shuffle": shuffle,
    "teacher_forcing_ratio": teacher_forcing_ratio,
}

suffix_data_split_value = 4

use_gradnorm = True
gn_alpha = 1.5
gn_learning_rate = 1e-4
number_tasks = len(dec_feat[0]) + len(dec_feat[1])

gradNorm = {
    "use_gradnorm": use_gradnorm,
    "number_tasks": number_tasks,
    "gn_alpha": gn_alpha,
    "gn_learning_rate": gn_learning_rate,
}

saving_path = os.path.join(HERE, f"BPIC_2017_remote_run_{timestamp}.pkl")
print(f"# checkpoints: {saving_path} (final) + _epoch<N>.pkl per epoch")

trainer = Trainer(
    device=device,
    model=model,
    data_train=BPIC_17_train_dataset,
    data_val=BPIC_17_val_dataset,
    loss_obj=loss_obj,
    log_normal_loss_num_feature=[],
    optimize_values=optimize_values,
    suffix_data_split_value=suffix_data_split_value,
    writer=writer,
    gradnorm_values=gradNorm,
    save_model_n_th_epoch=1,
    saving_path=saving_path,
)

train_attenuated_losses, val_losses, val_attenuated_losses = trainer.train_model()


summary_path = os.path.join(HERE, f"remote_training_losses_{timestamp}.txt")
with open(summary_path, "w") as f:
    f.write("epoch\ttrain_attenuated\tval_standard\tval_attenuated\n")
    for i, (ta, vs, va) in enumerate(
        zip(train_attenuated_losses, val_losses, val_attenuated_losses), start=1
    ):
        f.write(f"{i}\t{ta:.6f}\t{vs:.6f}\t{va:.6f}\n")
print(f"# Per-epoch losses written to {summary_path}")


fig_path = os.path.join(HERE, f"remote_training_loss_{timestamp}.png")
plt.figure()
epochs_axis = range(1, len(train_attenuated_losses) + 1)
plt.plot(epochs_axis, train_attenuated_losses, label='Training Attenuated Loss')
plt.plot(epochs_axis, val_losses, label='Validation Loss')
plt.plot(epochs_axis, val_attenuated_losses, label='Validation Attenuated Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"# Loss curve saved to {fig_path}")

print("# Done.")
