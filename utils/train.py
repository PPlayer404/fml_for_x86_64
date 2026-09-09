#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py —— 从标注数据训练车道线打分器权重（跨平台）

读取 annotate.py 标注后的 JSON（内嵌 24 维特征 + suggest 标注），
训练三种打分器并输出 weights.txt（fml_lane_scorer_load 可直接加载）：

  --mode m5    M5 乘法核（可解释档；网格搜索 a/b/cap/rank_w）
  --mode lr    白盒线性 LR（sklearn）
  --mode rbf   RBF-SVM（效果最强）

用法:
    python train.py <标注.json> [--mode rbf] [--out weights.txt]

依赖: pip install numpy scikit-learn
"""

import argparse
import json
import sys

import numpy as np


# 24 维特征顺序（与 extract_json / cluster.hpp 契约一致）
FEAT_NAMES = [
    "npts", "length", "score", "conf", "sumc", "theta", "absA1", "a2",
    "absA2yspan", "absA0dist", "ySpan", "cx_mean", "cx_std", "cx_range",
    "xTop", "xBot", "tilt_dxdy", "maxYgap", "nbGap", "ptDensity",
    "relScore", "relSumc", "rankNorm", "tiltAbs",
]


def load_samples(path, exclude_hard=True):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    X, y = [], []
    for im in data["images"]:
        for c in im["clusters"]:
            s = c.get("suggest", -1)
            if exclude_hard and s == 2:
                continue
            if s not in (0, 1):
                continue
            if len(c.get("features", [])) != 24:
                continue
            X.append(c["features"])
            y.append(1 if s == 1 else 0)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return data, X, y


# ---------------- M5 乘法核 ----------------

def m5_score(f, a, b, cap, rank_w, img_h=240.0):
    sumc = max(f[4], 0.0)
    lenw = f[1] / max(1.0, 0.5 * img_h)
    capp = min(lenw, cap)
    if capp <= 0.0:
        capp = 0.0
    logc = np.log10(1.0 + sumc)
    if logc <= 0.0:
        return -rank_w * max(0.0, f[22] - 1.0 / 3.0)
    s = np.power(logc, a) * np.power(capp, b)
    s -= rank_w * max(0.0, f[22] - 1.0 / 3.0)
    return s


def train_m5(X, y, img_h=240.0):
    from sklearn.metrics import roc_auc_score
    best = None
    n = len(y)
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        for b in (1.0, 1.2, 1.3, 1.5):
            for cap in (0.5, 0.6, 0.7, 0.8, 1.0):
                for rw in (0.1, 0.3, 0.5):
                    S = np.array([m5_score(f, a, b, cap, rw, img_h)
                                  for f in X])
                    if np.isnan(S).any() or np.isinf(S).any():
                        continue
                    # 交叉验证 AUC：样本充足时 5 折，样本少时留一折
                    rng = np.random.RandomState(0)
                    order = rng.permutation(n)
                    Xs, ys = X[order], y[order]
                    nfold = min(5, max(2, n // 8))
                    folds = np.array_split(np.arange(n), nfold)
                    aucs = []
                    for k in range(nfold):
                        te = folds[k]
                        if ys[te].min() == ys[te].max():
                            continue  # 单类折无法算 AUC，跳过
                        Ste = np.array([m5_score(f, a, b, cap, rw, img_h)
                                        for f in Xs[te]])
                        aucs.append(abs(roc_auc_score(ys[te], Ste) - 0.5) + 0.5)
                    if not aucs:
                        continue
                    auc = float(np.mean(aucs))
                    if best is None or auc > best[0]:
                        best = (auc, a, b, cap, rw)
    return best


def write_m5(weights, out):
    with open(out, "w", encoding="utf-8") as f:
        f.write("mode 1\n")
        f.write("img_h 240.0\n")
        f.write("m5_a %.6f\n" % weights[1])
        f.write("m5_b %.6f\n" % weights[2])
        f.write("m5_cap %.6f\n" % weights[3])
        f.write("m5_rank_w %.6f\n" % weights[4])
    return out


# ---------------- LR / RBF（sklearn） ----------------

def train_ml(X, y, mode, out):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    if mode == "lr":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=5000, class_weight="balanced")
    else:
        from sklearn.svm import SVC
        est = SVC(C=3, gamma="scale", kernel="rbf",
                  class_weight="balanced", probability=True)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # 交叉验证 AUC（单类折跳过；样本过小时退化为单一 holdout 汇报）
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    aucs = []
    if len(np.unique(y)) >= 2:
        try:
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            splits = list(skf.split(Xs, y))
        except ValueError:
            splits = [(np.arange(len(Xs)), np.arange(len(Xs)))]  # 近似holdout, 下方补齐
            splits = []
        if splits:
            for tr, te in splits:
                if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
                    continue
                m = est.fit(Xs[tr], y[tr])
                S = (m.predict_proba(Xs[te])[:, 1]
                     if hasattr(m, "predict_proba")
                     else m.decision_function(Xs[te]))
                aucs.append(roc_auc_score(y[te], S))
        if not aucs:  # 全部折单类 → 内联分层 holdout
            rng = np.random.RandomState(42)
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            n_te_p = max(1, int(0.3 * len(pos)))
            n_te_n = max(1, int(0.3 * len(neg)))
            te_p = rng.choice(pos, n_te_p, replace=False)
            te_n = rng.choice(neg, n_te_n, replace=False)
            te = np.concatenate([te_p, te_n])
            tr = np.setdiff1d(np.arange(len(y)), te)
            m = est.fit(Xs[tr], y[tr])
            S = (m.predict_proba(Xs[te])[:, 1]
                 if hasattr(m, "predict_proba") else m.decision_function(Xs[te]))
            aucs.append(roc_auc_score(y[te], S))
    auc = float(np.mean(aucs)) if aucs else float("nan")

    # 全量重训（最终权重）
    est.fit(Xs, y)

    if mode == "lr":
        w = est.coef_[0]
        b = float(est.intercept_[0])
        write_linear(out, sc.mean_, sc.scale_, w, b)
    else:
        sv = est.support_vectors_          # (n_sv × 24) 标准化空间
        coef = est.dual_coef_[0]           # (n_sv)
        gamma = float(est._gamma)
        b = float(est.intercept_[0])
        write_rbf(out, sc.mean_, sc.scale_, sv, coef, gamma, b)
    return auc


def fmt_arr(a):
    return " ".join("%.7g" % v for v in a)


def write_linear(out, mu, sig, w, b):
    with open(out, "w", encoding="utf-8") as f:
        f.write("mode 2\n")
        f.write("mu " + fmt_arr(mu) + "\n")
        f.write("istd " + fmt_arr(1.0 / sig) + "\n")
        f.write("w " + fmt_arr(w) + "\n")
        f.write("b %.7g\n" % b)


def write_rbf(out, mu, sig, sv, coef, gamma, b):
    with open(out, "w", encoding="utf-8") as f:
        f.write("mode 3\n")
        f.write("gamma %.8g\n" % gamma)
        f.write("b %.7g\n" % b)
        f.write("nsv %d\n" % len(coef))
        f.write("mu " + fmt_arr(mu) + "\n")
        f.write("istd " + fmt_arr(1.0 / sig) + "\n")
        for row in sv:
            f.write("sv " + fmt_arr(row) + "\n")
        f.write("svc " + fmt_arr(coef) + "\n")


def main():
    ap = argparse.ArgumentParser(description="车道线打分器权重训练")
    ap.add_argument("json", help="annotate.py 标注后的 JSON")
    ap.add_argument("--mode", choices=["m5", "lr", "rbf"], default="m5")
    ap.add_argument("--out", default="weights.txt")
    ap.add_argument("--img_h", type=float, default=240.0)
    args = ap.parse_args()

    data, X, y = load_samples(args.json)
    if len(y) < 20:
        print("样本太少（%d 条，需 >=20），请先标注更多数据" % len(y))
        return 1
    n_yes = int((y == 1).sum())
    n_no = int((y == 0).sum())
    print("样本: %d 条 (是=%d 否=%d)" % (len(y), n_yes, n_no))

    if args.mode == "m5":
        auc, a, b, cap, rw = train_m5(X, y, args.img_h)
        write_m5((auc, a, b, cap, rw), args.out)
        print("M5: a=%.2f b=%.2f cap=%.2f rank_w=%.2f  AUC=%.4f" %
              (a, b, cap, rw, auc))
    else:
        auc = train_ml(X, y, args.mode, args.out)
        print("%s 训练完成 AUC=%.4f" % (args.mode.upper(), auc))

    print("权重组 -> %s （直接用 fml_lane_scorer_load 加载）" % args.out)
    # 输出建议档位与说明
    print("加载: fml_cluster_params cp; cp.lane_scorer = %d; cp.lane_scorer_w = &w;"
          % {"m5": 1, "lr": 2, "rbf": 3}[args.mode])
    return 0


if __name__ == "__main__":
    sys.exit(main())