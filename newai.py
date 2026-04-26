#!/usr/bin/env python3
"""
train_wind_nn.py — Train wind danger prediction neural network
Requirements: pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json, os
# ── Configuration ────────────────────────────────────────────────────
EPOCHS      = 150
LR          = 0.01
HIDDEN1     = 12
HIDDEN2     = 8
CLASSES     = 5
BATCH_SIZE  = 32
LABELS      = ["SAFE","MODERATE","DANGER","HIGH_DANGER","EXTREME"]
# ── Load Dataset ─────────────────────────────────────────────────────
df = pd.read_csv("wind_dataset.csv")
X  = df[["speed_norm","gust_norm","dir_norm","stddev_norm",
          "roc_norm","humidity_norm","pressure_norm","temp_norm"]].values
y  = df["label"].values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print(f"Class distribution: {np.bincount(y)}")
# ── Weight Initialization (He) ────────────────────────────────────────
def init_w(i, o): return np.random.randn(o, i) * np.sqrt(2 / i)
def init_b(n):    return np.zeros(n) + 0.01
np.random.seed(42)
W1 = init_w(8, HIDDEN1); b1 = init_b(HIDDEN1)
W2 = init_w(HIDDEN1, HIDDEN2); b2 = init_b(HIDDEN2)
W3 = init_w(HIDDEN2, CLASSES); b3 = init_b(CLASSES)
# ── Activation Functions ──────────────────────────────────────────────
relu    = lambda x: np.maximum(0, x)
d_relu  = lambda x: (x > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def cross_entropy(p, y):
    return -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
# ── Forward Pass ──────────────────────────────────────────────────────
def forward(X):
    H1  = relu(X @ W1.T + b1)
    H2  = relu(H1 @ W2.T + b2)
    out = softmax(H2 @ W3.T + b3)
    return H1, H2, out
# ── Training Loop ─────────────────────────────────────────────────────
history = {"loss":[], "val_loss":[], "acc":[], "val_acc":[]}
best_acc = 0
for epoch in range(EPOCHS):
    # Mini-batch SGD
    idx = np.random.permutation(len(X_train))
    for start in range(0, len(X_train), BATCH_SIZE):
        batch = idx[start:start+BATCH_SIZE]
        Xb, yb = X_train[batch], y_train[batch]
        # Forward
        H1, H2, out = forward(Xb)
        # Backward (cross-entropy + softmax combined gradient)
        dOut = out.copy()
        dOut[np.arange(len(yb)), yb] -= 1
        dOut /= len(yb)
        dW3 = H2.T @ dOut;   db3 = dOut.sum(0)
        dH2 = dOut @ W3;     dH2 *= d_relu(H2)
        dW2 = H1.T @ dH2;   db2 = dH2.sum(0)
        dH1 = dH2 @ W2;     dH1 *= d_relu(H1)
        dW1 = Xb.T @ dH1;   db1_ = dH1.sum(0)
        # Gradient clipping
        clip = lambda g: np.clip(g, -5, 5)
        W3 -= LR * clip(dW3.T); b3 -= LR * clip(db3)
        W2 -= LR * clip(dW2.T); b2 -= LR * clip(db2)
        W1 -= LR * clip(dW1.T); b1 -= LR * clip(db1_)
    # Epoch metrics
    _, _, train_out = forward(X_train)
    _, _, val_out   = forward(X_test)
    tloss = cross_entropy(train_out, y_train)
    vloss = cross_entropy(val_out,   y_test)
    tacc  = (train_out.argmax(1) == y_train).mean()
    vacc  = (val_out.argmax(1)   == y_test).mean()
    history["loss"].append(tloss)
    history["val_loss"].append(vloss)
    history["acc"].append(tacc)
    history["val_acc"].append(vacc)
    if vacc > best_acc:
        best_acc = vacc
        np.savez("best_weights.npz",
                 W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: loss={tloss:.4f} "
              f"val_loss={vloss:.4f} "
              f"val_acc={vacc:.1%}")
# ── Final Evaluation ──────────────────────────────────────────────────
best = np.load("best_weights.npz")
W1,b1,W2,b2,W3,b3 = (best[k] for k in
                      ["W1","b1","W2","b2","W3","b3"])
_, _, final = forward(X_test)
preds = final.argmax(1)
print("\n" + "="*50)
print(f"Best Validation Accuracy: {best_acc:.1%}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=LABELS))
# ── Export Weights as C Header ────────────────────────────────────────
def to_c_array(arr, name):
    flat = arr.flatten()
    vals = ", ".join(f"{v:.6f}f" for v in flat)
    return f"const float {name}[] = {{\n  {vals}\n}};"
with open("nn_weights.h", "w") as f:
    f.write("// Auto-generated — paste into ESP32 firmware\n")
    f.write(f"// Architecture: 8->{HIDDEN1}->{HIDDEN2}->{CLASSES}\n\n")
    f.write(to_c_array(W1, "W1") + "\n\n")
    f.write(to_c_array(b1, "b1") + "\n\n")
    f.write(to_c_array(W2, "W2") + "\n\n")
    f.write(to_c_array(b2, "b2") + "\n\n")
    f.write(to_c_array(W3, "W3") + "\n\n")
    f.write(to_c_array(b3, "b3") + "\n")
print("\nWeights exported to nn_weights.h")
print("Copy nn_weights.h into your Arduino project folder.")