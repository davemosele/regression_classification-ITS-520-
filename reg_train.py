import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

torch.manual_seed(42)

class TabularRegressor(nn.Module):
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

def train_loop(model, Xtr, ytr, Xva, yva, lr=1e-3, epochs=80, bs=128):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = Xtr.shape[0]
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb = torch.from_numpy(Xtr[idx]).float()
            yb = torch.from_numpy(ytr[idx]).float().unsqueeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                pv = model(torch.from_numpy(Xva).float()).squeeze(1).numpy()
            r2 = r2_score(yva, pv)
            print(f"[ep {ep:03d}] train_mse={total/n:.4f}  val_R2={r2:.4f}")
    return model

def eval_and_report(model, Xtr, ytr, Xte, yte, name):
    model.eval()
    with torch.no_grad():
        p_tr = model(torch.from_numpy(Xtr).float()).squeeze(1).numpy()
        p_te = model(torch.from_numpy(Xte).float()).squeeze(1).numpy()
    r2_tr = r2_score(ytr, p_tr); r2_te = r2_score(yte, p_te)
    mse_te = mean_squared_error(yte, p_te); mae_te = mean_absolute_error(yte, p_te)
    print(f"\n[{name}] R2 train={r2_tr:.4f}  R2 test={r2_te:.4f}  MSE test={mse_te:.4f}  MAE test={mae_te:.4f}")
    return {"r2_train": r2_tr, "r2_test": r2_te, "mse_test": mse_te, "mae_test": mae_te}

def export_onnx(model, scaler, feature_names, out_path="reg_model_v2.onnx"):
    model.eval()
    in_dim = len(feature_names)
    dummy = torch.randn(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["pred"],
        dynamic_axes={"input": {0: "batch"}, "pred": {0: "batch"}},
        opset_version=17,
        use_external_data_format=False
    )
    prep = {
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    Path("reg_preproc.json").write_text(json.dumps(prep, indent=2))
    print(f"Exported ONNX → {out_path} and reg_preproc.json")

def main():
    data = load_diabetes()
    X = data.data.astype(np.float32)        # 10 features
    y = data.target.astype(np.float32)      # disease progression
    feature_names = list(data.feature_names)  # ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train); X_valid = scaler.transform(X_valid); X_test = scaler.transform(X_test)

    linear = TabularRegressor(X.shape[1], kind="linear")
    mlp = TabularRegressor(X.shape[1], kind="mlp")

    print("\n== Train: Linear ==")
    linear = train_loop(linear, X_train, y_train, X_valid, y_valid, lr=1e-2, epochs=80)
    m_lin = eval_and_report(linear, X_train, y_train, X_test, y_test, name="Linear")

    print("\n== Train: MLP ==")
    mlp = train_loop(mlp, X_train, y_train, X_valid, y_valid, lr=3e-3, epochs=80)
    m_mlp = eval_and_report(mlp, X_train, y_train, X_test, y_test, name="MLP")

    best, best_name = (mlp, "MLP") if m_mlp["r2_test"] >= m_lin["r2_test"] else (linear, "Linear")
    print(f"\nBest regression model: {best_name}")
    export_onnx(best, scaler, feature_names, out_path="reg_model_v2.onnx")

if __name__ == "__main__":
    main()
