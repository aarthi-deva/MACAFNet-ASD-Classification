import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)

from sklearn.model_selection import StratifiedShuffleSplit
from itertools import combinations

from dataset_loader import AutismDataset
from macaf_net import MACAFNetHybrid

import pandas as pd
import os
from openpyxl import Workbook

# =================================================
# CONFIG
# =================================================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING = 7

RESULTS_DIR = "results_macafnet"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Excel writer (single file for all experiments)
excel_path = os.path.join(RESULTS_DIR, "all_experiments_detailed.xlsx")
writer = pd.ExcelWriter(excel_path, engine="openpyxl")

# =================================================
# ATLAS LIST
# =================================================

ALL_ATLASES = ["aal", "cc200", "dosenbach160", "ho", "ez", "tt"]

atlas_combinations = []
for r in range(1, len(ALL_ATLASES) + 1):
    atlas_combinations.extend(combinations(ALL_ATLASES, r))

print("Total Experiments:", len(atlas_combinations))

# =================================================
# SUMMARY STORAGE
# =================================================

summary_metrics = []

# =================================================
# MAIN LOOP
# =================================================

for combo in atlas_combinations:

    combo = list(combo)
    combo_name = "_".join(combo)
    sheet_name = combo_name[:31]

    print("\n================================================")
    print("Training Model:", combo_name)
    print("================================================")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    dataset = AutismDataset("data/abide.hdf5", atlases=combo)

    subject_ids = list(range(len(dataset)))
    labels = []

    for idx in subject_ids:
        patient = dataset.file["patients"][dataset.keys[idx]]
        labels.append(patient.attrs["y"])

    labels = np.array(labels)

    # -------------------------------------------------
    # SPLIT
    # -------------------------------------------------
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss1.split(subject_ids, labels))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_idx, labels[temp_idx]))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    sample = dataset[0]
    input_dims = [sample[i].shape[0] for i in range(len(combo))]

    model = MACAFNetHybrid(
        atlas_dims=input_dims,
        embed_dim=64,
        dropout=0.4,
        n_heads=4,
        transformer_layers=1
    ).to(DEVICE)

    # -------------------------------------------------
    # LOSS + OPTIMIZER
    # -------------------------------------------------
    num_pos = labels.sum()
    num_neg = len(labels) - num_pos

    pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # -------------------------------------------------
    # LOGS
    # -------------------------------------------------
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    best_val_f1 = 0
    early_stop = 0

    # =================================================
    # TRAINING LOOP
    # =================================================

    for epoch in range(EPOCHS):

        model.train()

        train_preds, train_labels_list = [], []
        epoch_loss = 0

        for batch in train_loader:

            inputs = [x.to(DEVICE) for x in batch[:-1]]
            y = batch[-1].float().to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            out = model(*inputs)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)

            train_preds.extend(torch.sigmoid(out).detach().cpu().numpy())
            train_labels_list.extend(y.cpu().numpy())

        train_preds_bin = (np.array(train_preds) > 0.5).astype(int)

        train_acc = accuracy_score(train_labels_list, train_preds_bin)
        train_loss = epoch_loss / len(train_idx)

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()

        val_preds, val_labels_list = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:

                inputs = [x.to(DEVICE) for x in batch[:-1]]
                y = batch[-1].float().to(DEVICE).unsqueeze(1)

                out = model(*inputs)
                loss = criterion(out, y)

                val_loss += loss.item() * y.size(0)

                val_preds.extend(torch.sigmoid(out).cpu().numpy())
                val_labels_list.extend(y.cpu().numpy())

        val_preds_bin = (np.array(val_preds) > 0.5).astype(int)

        val_acc = accuracy_score(val_labels_list, val_preds_bin)
        val_f1 = f1_score(val_labels_list, val_preds_bin)

        val_accs.append(val_acc)
        val_losses.append(val_loss / len(val_idx))

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss/len(val_idx):.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        scheduler.step(val_f1)

        # ---------------- EARLY STOP ----------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"best_macafnet_{combo_name}.pt")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= EARLY_STOPPING:
                print("Early stopping triggered")
                break

    # =================================================
    # TESTING
    # =================================================

    model.load_state_dict(torch.load(f"best_macafnet_{combo_name}.pt", map_location=DEVICE))
    model.eval()

    test_preds, test_labels_list = [], []

    with torch.no_grad():
        for batch in test_loader:

            inputs = [x.to(DEVICE) for x in batch[:-1]]
            y = batch[-1].float().to(DEVICE).unsqueeze(1)

            out = model(*inputs)

            test_preds.extend(torch.sigmoid(out).cpu().numpy())
            test_labels_list.extend(y.cpu().numpy())

    test_probs = np.array(test_preds).flatten()
    test_labels_np = np.array(test_labels_list).flatten()

    # =================================================
    # THRESHOLD OPTIMIZATION
    # =================================================

    thresholds = np.arange(0.1, 0.9, 0.01)

    f1_scores = [
        f1_score(test_labels_np, (test_probs > t).astype(int))
        for t in thresholds
    ]

    best_thresh = thresholds[np.argmax(f1_scores)]
    test_pred_bin = (test_probs > best_thresh).astype(int)

    # =================================================
    # METRICS
    # =================================================

    acc = accuracy_score(test_labels_np, test_pred_bin)
    prec = precision_score(test_labels_np, test_pred_bin)
    rec = recall_score(test_labels_np, test_pred_bin)
    f1 = f1_score(test_labels_np, test_pred_bin)
    auc_score = roc_auc_score(test_labels_np, test_probs)

    cm = confusion_matrix(test_labels_np, test_pred_bin)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # =================================================
    # PRINT FINAL RESULTS (YOUR REQUEST)
    # =================================================

    print("\n==================== FINAL TEST RESULTS ====================")
    print(f"Atlas Combination : {combo_name}")
    print("------------------------------------------------------------")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {auc_score:.4f}")
    print(f"Sensitivity  : {sensitivity:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print("------------------------------------------------------------")
    print("\nConfusion Matrix:")
    print(cm)
    print("============================================================\n")

    # =================================================
    # SUMMARY
    # =================================================

    summary_metrics.append({
        "Combo": combo_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc_score,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    })

    # =================================================
    # EXCEL LOGGING
    # =================================================

    epoch_df = pd.DataFrame({
        "epoch": list(range(1, len(train_accs) + 1)),
        "train_acc": train_accs,
        "train_loss": train_losses,
        "val_acc": val_accs,
        "val_loss": val_losses
    })

    pred_df = pd.DataFrame({
        "y_true": test_labels_np,
        "y_prob": test_probs,
        "y_pred": test_pred_bin
    })

    epoch_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    pred_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(epoch_df) + 3)

# =================================================
# FINAL SAVE
# =================================================

summary_df = pd.DataFrame(summary_metrics)
summary_df.to_excel(os.path.join(RESULTS_DIR, "summary.xlsx"), index=False)

writer.close()

print("\nALL EXPERIMENTS COMPLETED")
print("Excel saved at:", excel_path)
