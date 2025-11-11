import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

torch.manual_seed(42)

class TabularClassifier(nn.Module):
    def __init__(self, in_dim, kind="mlp"):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.Linear(in_dim, 1))
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
    def forward(self, x): return self.net(x)

def train_loop(model, Xtr, ytr, Xva, yva, lr=1e-3, epochs=45, bs=128):
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb = torch.from_numpy(Xtr[idx]).float()
            yb = torch.from_numpy(ytr[idx]).float().unsqueeze(1)
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        if ep % 7 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                lv = model(torch.from_numpy(Xva).float()).squeeze(1)
                pv = (torch.sigmoid(lv).numpy() >= 0.5).astype(np.int64)
            acc = (pv == yva).mean()
            print(f"[ep {ep:03d}] train_loss={tot/n:.4f}  val_acc={acc:.4f}")
    return model

def eval_and_report(model, Xtr, ytr, Xte, yte, name):
    model.eval()
    with torch.no_grad():
        lt = model(torch.from_numpy(Xte).float()).squeeze(1)
        pt = (torch.sigmoid(lt).numpy() >= 0.5).astype(np.int64)
    acc = accuracy_score(yte, pt)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pt, average="binary", zero_division=0)
    cm = confusion_matrix(yte, pt)
    print(f"\n[{name}] acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    print("Confusion matrix:\n", cm)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm.tolist()}

def export_onnx(model, scaler, feature_names, out_path="clf_model.onnx"):
    model.eval()
    in_dim = len(feature_names)
    dummy = torch.randn(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )
    prep = {
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    Path("clf_preproc.json").write_text(json.dumps(prep, indent=2))
    print(f"Exported ONNX → {out_path} and clf_preproc.json")

def main():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)    # 30 features
    y = data.target.astype(np.int64)    # 0/1 labels
    feature_names = list(data.feature_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train); X_valid = scaler.transform(X_valid); X_test = scaler.transform(X_test)

    linear = TabularClassifier(X.shape[1], kind="linear")
    mlp = TabularClassifier(X.shape[1], kind="mlp")

    print("\n== Train: Linear ==")
    linear = train_loop(linear, X_train, y_train, X_valid, y_valid, lr=1e-2, epochs=45)
    m_lin = eval_and_report(linear, X_train, y_train, X_test, y_test, name="Linear")

    print("\n== Train: MLP ==")
    mlp = train_loop(mlp, X_train, y_train, X_valid, y_valid, lr=3e-3, epochs=45)
    m_mlp = eval_and_report(mlp, X_train, y_train, X_test, y_test, name="MLP")

    best, best_name = (mlp, "MLP") if m_mlp["f1"] >= m_lin["f1"] else (linear, "Linear")
    print(f"\nBest classification model: {best_name}")
    export_onnx(best, scaler, feature_names, out_path="clf_model.onnx")

if __name__ == "__main__":
    main()
