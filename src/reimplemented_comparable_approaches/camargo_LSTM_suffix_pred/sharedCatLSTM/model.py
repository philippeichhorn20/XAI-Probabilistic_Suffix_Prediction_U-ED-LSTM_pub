"""Shared-categorical LSTM from Camargo, Dumas & González-Rojas (2019).

Paper Fig. 6(b): the shared LSTM only sees categorical (embedded) inputs. Numeric
features bypass the shared layer and are injected directly into the per-head
specialised LSTM. Compared to `joinLSTM.model.FullShared_Join_LSTM`, this keeps
categorical-vs-numerical signal in separate processing lanes up to the specialised
head — Table 3 of the paper shows this is the best-performing variant on Helpdesk
(DL act sim 0.9568 vs 0.5773 for the full-shared variant).

Input/output interface matches `FullShared_Join_LSTM`:
    forward(input=(cats, nums)) -> raw logits [B, n_activity_classes]
    model((cats, nums))  # cats: list of [B, T] int tensors, nums: list of [B, T] float tensors
Apply `F.softmax` at call sites that need a probability distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCat_LSTM(nn.Module):
    def __init__(
        self,
        data_set_categories: list,
        hidden_size: int,
        num_layers: int,
        model_feat: list,
        input_size: int,
        output_size_act: int,
    ):
        super().__init__()

        self.data_set_categories = data_set_categories
        print("Data set categories: ", data_set_categories)

        self.model_feat = model_feat
        print("Model input features: ", model_feat)

        cat_categories, _ = data_set_categories
        cat_input_feat_model, num_input_feat_model = model_feat

        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        list_of_classes_per_cat = [cat_dict[c] for c in cat_input_feat_model if c in cat_dict]

        # Embeddings for categorical features (same sizing rule as joinLSTM)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in list_of_classes_per_cat]
        )
        print("Embeddings: ", self.embeddings)

        embedding_size = sum(min(600, round(1.6 * n_cat**0.56)) for n_cat in list_of_classes_per_cat)
        print("Total embedding feature size: ", embedding_size)

        self.num_features = len(num_input_feat_model)
        print("Number of numerical features: ", self.num_features)

        # input_size==1 sentinel means "compute from scratch"; otherwise restore a saved size
        if input_size == 1:
            self.input_size = embedding_size  # shared LSTM only reads cat embeddings
        else:
            self.input_size = input_size
        print("Input feature size (shared LSTM, cat-only): ", self.input_size)

        self.hidden_size = hidden_size
        print("Cells hidden size: ", hidden_size)

        self.num_layers = num_layers
        print("Number of LSTM layer: ", num_layers)

        self.output_size_act = output_size_act

        # Shared LSTM — categorical inputs only.
        self.shared_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=0.2,
            num_layers=self.num_layers,
        )

        # Batch-norm across features per time step.
        self.bn1 = nn.BatchNorm1d(self.hidden_size)

        # Head LSTM sees shared_hidden ⧺ raw numeric features.
        self.lstm_act = nn.LSTM(
            input_size=self.hidden_size + self.num_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=0.2,
            num_layers=self.num_layers,
        )

        # Linear output head.
        self.act_head = nn.Linear(self.hidden_size, self.output_size_act)

    def __embed_cats(self, cats):
        embedded = [emb(cats[i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=-1)  # [B, T, sum_embed]

    def __stack_nums(self, nums, device):
        if not nums:
            # nothing to concatenate; caller will just use shared output as-is
            return None
        return torch.stack(nums, dim=-1).float().to(device)  # [B, T, n_num]

    def forward(self, input):
        cats, nums = input
        # [B, T, sum_embed]
        cat_embed = self.__embed_cats(cats)
        num_stack = self.__stack_nums(nums, cat_embed.device)  # [B, T, n_num] or None

        # Shared LSTM over categorical embeddings only.
        out_shared, _ = self.shared_lstm(cat_embed)  # [B, T, H]

        # Batch-norm per time step.
        y = out_shared.transpose(1, 2)  # [B, H, T]
        y = self.bn1(y)
        y = y.transpose(1, 2)  # [B, T, H]

        # Concatenate numerical features into the head's input.
        head_input = torch.cat([y, num_stack], dim=-1) if num_stack is not None else y

        # Specialised head LSTM — returns last hidden state.
        _, (h_act, _) = self.lstm_act(head_input)  # h_act: [1, B, H]
        h_act = h_act.squeeze(0)  # [B, H]

        # Return raw logits so F.cross_entropy can do log_softmax internally.
        # Callers that need a probability distribution must apply F.softmax themselves.
        a_logits = self.act_head(h_act)
        return a_logits

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": {
                "data_set_categories": self.data_set_categories,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "model_feat": self.model_feat,
                "input_size": self.input_size,
                "output_size_act": self.output_size_act,
            },
        }
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        model = SharedCat_LSTM(**checkpoint["kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
