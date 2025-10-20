#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, glob, argparse, random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# ---- 全局 DoG 开关（保证即便没装 skimage 也不会 NameError）----
NO_DOG = False

# 可选：scikit-image 用于 DoG blob 更稳
try:
    from skimage.feature import blob_dog
    SKIMG_OK = True
except Exception:
    SKIMG_OK = False

from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from joblib import Parallel, delayed

# -----------------------------
# 频域 / JPEG / 轻量去噪 工具
# -----------------------------
from math import sqrt

def psd_ring_features(gray, n_rings=20):
    """返回: (ring_ratio, peakiness) 两维，用于构成随 α 的曲线。"""
    g = gray.astype(np.float32)
    h, w = g.shape
    # Hann 窗抑制泄露
    wy = np.hanning(h)[:, None]; wx = np.hanning(w)[None, :]
    win = (wy * wx).astype(np.float32)
    G = np.fft.fftshift(np.fft.fft2(g * win))
    P = np.abs(G) ** 2

    cy, cx = h // 2, w // 2
    rmax = sqrt((cy ** 2) + (cx ** 2))
    rings = np.linspace(1, rmax, n_rings + 1)
    yy, xx = np.ogrid[:h, :w]  # 预计算
    eng = []
    for i in range(n_rings):
        r0, r1 = rings[i], rings[i + 1]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        m = (r >= r0) & (r < r1)
        e = P[m].mean() if np.any(m) else 0.0
        eng.append(e + 1e-9)
    eng = np.array(eng, dtype=np.float64)
    ring_ratio = float(np.mean(eng[int(n_rings * 0.25):int(n_rings * 0.6)]) / np.mean(eng))
    peakiness = float(np.max(eng) / (np.median(eng)))
    return ring_ratio, peakiness

def jpeg_reencode_metrics(gray, quality):
    """返回 ΔMSE, 1-SSIM, 中频能量比（重编码后）"""
    ok, enc = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return 0.0, 0.0, 0.0
    rec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if rec is None:
        return 0.0, 0.0, 0.0
    diff = (gray.astype(np.float32) - rec.astype(np.float32))
    dmse = float(np.mean(diff ** 2))
    # 简版 SSIM
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(gray.astype(np.float32), (7, 7), 1.5)
    mu2 = cv2.GaussianBlur(rec.astype(np.float32), (7, 7), 1.5)
    s1 = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (7, 7), 1.5) - mu1 ** 2
    s2 = cv2.GaussianBlur(rec.astype(np.float32) ** 2, (7, 7), 1.5) - mu2 ** 2
    s12 = cv2.GaussianBlur((gray.astype(np.float32) * rec.astype(np.float32)), (7, 7), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2) + 1e-9)
    ssim = float(np.clip(np.mean(ssim_map), 0, 1))
    # 中频能量比
    _, mid = dct_zero_run_and_midratio(rec)
    return dmse, (1.0 - ssim), mid

def fast_denoise(gray):
    """近似外部去噪器（快）：双边 + 轻高斯"""
    out = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=5)
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    return out

# -----------------------------
# 读图 / 加噪 / 基础指标
# -----------------------------
def imread_gray(path, max_size=512):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = max(h, w)
    if m > max_size:
        scale = max_size / m
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray

def add_noise(gray, noise_type, alpha, rng):
    """alpha ∈ [0,1]，控制噪声强度（相对图像动态范围 0..255）"""
    if alpha is None or alpha == 0:
        return gray.copy()
    g = gray.astype(np.float32)
    if noise_type == "gaussian":
        sigma = alpha * 10.0
        n = rng.normal(0, sigma, size=g.shape).astype(np.float32)
        out = g + n
    elif noise_type == "poisson":
        lam = np.clip(g / 255.0, 0, 1.0) * (alpha * 30.0 + 1.0)
        out = rng.poisson(lam).astype(np.float32)  # 用同一 RNG，保证可复现
        out = out / (alpha * 30.0 + 1.0) * 255.0
    elif noise_type == "quant":
        b = int(8 - round(alpha * 3))
        b = max(4, min(8, b))
        levels = 2 ** b
        step = 255.0 / (levels - 1)
        q = np.round(g / step) * step
        dither = rng.uniform(-0.5 * step, 0.5 * step, size=g.shape).astype(np.float32)
        out = q + dither
    else:
        out = g
    return np.clip(out, 0, 255).astype(np.uint8)

def grad_entropy(gray, num_bins=64):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mmax = float(mag.max())
    if not np.isfinite(mmax) or mmax < 1e-6:
        return 0.0
    hist, _ = np.histogram(mag.ravel(), bins=num_bins, range=(0, mmax + 1e-6), density=True)
    hist = np.clip(hist, 1e-12, None)
    return float(entropy(hist, base=2))

def dog_kp_count(gray):
    if NO_DOG:
        return 1  # 让 DoG 留存率≈1，避免影响其它指标
    if SKIMG_OK:
        blobs = blob_dog(gray.astype(np.float32) / 255.0, min_sigma=1.0, max_sigma=5.0, threshold=0.02)
        return int(len(blobs))
    else:
        g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        g2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
        dog = cv2.absdiff(g1, g2)
        thr = max(10, int(dog.mean() + 2 * dog.std()))
        _, bw = cv2.threshold(dog, thr, 255, cv2.THRESH_BINARY)
        n, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return int(len(n))

# -----------------------------
# 8x8 DCT 工具
# -----------------------------
ZIGZAG_IDX = np.array([
    [0, 1, 5, 6,14,15,27,28],
    [2, 4, 7,13,16,26,29,42],
    [3, 8,12,17,25,30,41,43],
    [9,11,18,24,31,40,44,53],
    [10,19,23,32,39,45,52,54],
    [20,22,33,38,46,51,55,60],
    [21,34,37,47,50,56,59,61],
    [35,36,48,49,57,58,62,63]
]).ravel()

def block_dct8x8(gray):
    h, w = gray.shape
    H = h - (h % 8)
    W = w - (w % 8)
    if H == 0 or W == 0:
        return np.empty((0, 8, 8), dtype=np.float32)
    img = gray[:H, :W].astype(np.float32) - 128.0
    blocks = []
    for y in range(0, H, 8):
        for x in range(0, W, 8):
            blk = img[y:y+8, x:x+8]
            c = cv2.dct(blk)
            blocks.append(c)
    return np.array(blocks, dtype=np.float32)

def dct_zero_run_and_midratio(gray):
    """返回：平均零游程长度(zigzag, 去掉DC)， 中频能量比(中频/总能量)"""
    blocks = block_dct8x8(gray)
    if blocks.size == 0:
        return 0.0, 0.0
    coeffs = blocks.reshape((-1, 64))
    coeffs = coeffs[:, ZIGZAG_IDX]
    ac = coeffs[:, 1:]
    runs = []
    for row in ac:
        zeros = (np.abs(row) < 1e-6).astype(np.int32)
        cnt = 0; seg = []
        for z in zeros:
            if z == 1:
                cnt += 1
            else:
                if cnt > 0: seg.append(cnt); cnt = 0
        if cnt > 0: seg.append(cnt)
        runs.append(float(np.mean(seg)) if seg else 0.0)
    mean_zero_run = float(np.mean(runs)) if runs else 0.0

    mid_mask = np.ones((8, 8), np.uint8)
    mid_mask[:3, :3] = 0
    mid_mask[-3:, -3:] = 0
    mid_mask = mid_mask.astype(bool).ravel()
    E_total = np.sum(blocks ** 2, axis=(1, 2)) + 1e-9
    E_mid   = np.sum((blocks.reshape((-1, 64)) ** 2)[:, mid_mask], axis=1)
    ratio = float(np.mean(E_mid / E_total))
    return mean_zero_run, ratio

def polyfit_quad(x, y):
    """返回 (a,b,c,r2)"""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if len(x) < 3:
        p = np.polyfit(x, y, 1)
        a, b, c = 0.0, p[0], p[1]
        yhat = b * x + c
    else:
        p = np.polyfit(x, y, 2)
        a, b, c = p[0], p[1], p[2]
        yhat = a * (x ** 2) + b * x + c
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(c), float(r2)

# -----------------------------
# 单图：多噪声、多 α → 指标 → 拟合曲线
# -----------------------------
def process_one_image(path, noises, alphas, rng_seed=1234, max_size=512):
    rng = np.random.default_rng(rng_seed + hash(path) % (10 ** 6))
    gray0 = imread_gray(path, max_size=max_size)
    x = np.array(alphas, dtype=np.float64)

    # DoG baseline
    try:
        base_kp = dog_kp_count(gray0)
    except Exception:
        base_kp = 1
    base_kp = max(1, base_kp)

    features = []
    for ntype in noises:
        ys_gradH, ys_kpret, ys_zrun, ys_mid = [], [], [], []
        for a in alphas:
            img = add_noise(gray0, ntype, a, rng)
            try: gH = grad_entropy(img)
            except Exception: gH = 0.0
            try: kp = dog_kp_count(img)
            except Exception: kp = base_kp
            kpret = max(0.0, float(kp) / float(base_kp))
            try: zrun, midr = dct_zero_run_and_midratio(img)
            except Exception: zrun, midr = 0.0, 0.0
            ys_gradH.append(float(gH))
            ys_kpret.append(float(kpret))
            ys_zrun.append(float(zrun))
            ys_mid.append(float(midr))

        for name, ys in [
            (f"{ntype}_gradH", ys_gradH),
            (f"{ntype}_DoGret", ys_kpret),
            (f"{ntype}_DCTzrun", ys_zrun),
            (f"{ntype}_DCTmid", ys_mid),
        ]:
            a,b,c,r2 = polyfit_quad(x, ys)
            features.append((name, (a,b,c,r2)))

        # A) 频域环能量
        ys_ring, ys_peak = [], []
        for a in alphas:
            img = add_noise(gray0, ntype, a, rng)
            try: rrat, pks = psd_ring_features(img)
            except Exception: rrat, pks = 0.0, 0.0
            ys_ring.append(rrat); ys_peak.append(pks)
        for name, ys in [(f"{ntype}_PSDring", ys_ring), (f"{ntype}_PSDpeak", ys_peak)]:
            a2,b2,c2,r2 = polyfit_quad(x, ys)
            features.append((name, (a2,b2,c2,r2)))

        # B) 再 JPEG 探针（固定 α=0.2 与 0.6）
        a_fixed_list = []
        if len(alphas) >= 2: a_fixed_list.append(alphas[1])     # 0.2
        if len(alphas) >= 3: a_fixed_list.append(alphas[-2])    # 0.6
        if not a_fixed_list: a_fixed_list = [0.4]
        for a_fixed in a_fixed_list:
            imgA = add_noise(gray0, ntype, a_fixed, rng)
            Qs = [95, 85, 75]
            ys_dmse, ys_dssim, ys_mid2 = [], [], []
            for Q in Qs:
                try: d1, d2, mid = jpeg_reencode_metrics(imgA, Q)
                except Exception: d1, d2, mid = 0.0, 0.0, 0.0
                ys_dmse.append(d1); ys_dssim.append(d2); ys_mid2.append(mid)
            qx = np.array(Qs, dtype=np.float64)
            for nm, ys in [(f"{ntype}_JPEGdmse_a{a_fixed}", ys_dmse),
                           (f"{ntype}_JPEGdssim_a{a_fixed}", ys_dssim),
                           (f"{ntype}_JPEGmid_a{a_fixed}", ys_mid2)]:
                a3,b3,c3,r23 = polyfit_quad(qx, ys)
                features.append((nm, (a3,b3,c3,r23)))

        # C) 去噪残差响应
        ys_resE, ys_resRing = [], []
        for a in alphas:
            noisy = add_noise(gray0, ntype, a, rng)
            try:
                den = fast_denoise(noisy)
                res = cv2.absdiff(noisy, den).astype(np.float32)
                resE = float(np.mean(res ** 2) / (np.mean(noisy.astype(np.float32) ** 2) + 1e-9))
                rrat, _ = psd_ring_features((res * 4).clip(0, 255).astype(np.uint8))
            except Exception:
                resE, rrat = 0.0, 0.0
            ys_resE.append(resE); ys_resRing.append(rrat)
        for nm, ys in [(f"{ntype}_ResEnergy", ys_resE),
                       (f"{ntype}_ResRing", ys_resRing)]:
            a4,b4,c4,r24 = polyfit_quad(x, ys)
            features.append((nm, (a4,b4,c4,r24)))

    # 拼接向量
    coef_vec, r2_vec, names = [], [], []
    for name, (a,b,c,r2) in features:
        coef_vec += [a,b,c]; r2_vec.append(r2); names.append(name)
    return np.array(coef_vec, dtype=np.float32), np.array(r2_vec, dtype=np.float32), names

# -----------------------------
# 数据集批处理 & 评测
# -----------------------------
def collect_image_paths(root, exts=(".jpg",".jpeg",".png",".bmp",".webp",".JPEG",".JPG",".PNG")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, f"*{e}"))
    print(f"[DEBUG] Scanned {root}, found {len(paths)} files")
    if len(paths) > 0:
        print("[DEBUG] Example files:", paths[:5])
    return sorted(paths)

def main_extract(args):
    global NO_DOG
    NO_DOG = args.no_dog

    real_paths = collect_image_paths(args.real_dir)
    fake_paths = collect_image_paths(args.fake_dir)

    if len(real_paths) == 0:
        print(f"[ERROR] No files found in real_dir={args.real_dir}")
    if len(fake_paths) == 0:
        print(f"[ERROR] No files found in fake_dir={args.fake_dir}")

    if args.max_images_per_class is not None and args.max_images_per_class > 0:
        random.seed(0)
        if len(real_paths) > 0:
            real_paths = random.sample(real_paths, min(args.max_images_per_class, len(real_paths)))
        if len(fake_paths) > 0:
            fake_paths = random.sample(fake_paths, min(args.max_images_per_class, len(fake_paths)))

    print(f"[INFO] #real={len(real_paths)}  #fake={len(fake_paths)}")
    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("[FATAL] One of the class dirs is empty. Exiting.")
        sys.exit(1)

    # 保存抽样列表（用于后续与 D3 对齐）
    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
    with open(args.out_csv + ".real.list.txt", "w") as f:
        for p in real_paths: f.write(p + "\n")
    with open(args.out_csv + ".fake.list.txt", "w") as f:
        for p in fake_paths: f.write(p + "\n")
    print(f"[INFO] saved file lists: {args.out_csv}.real.list.txt / .fake.list.txt")

    print("[INFO] extracting features ...")
    noises = [s.strip().lower() for s in args.noises.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    def _proc(pth, label, idx):
        try:
            coef, r2, names = process_one_image(
                pth, noises, alphas, rng_seed=418, max_size=args.max_size
            )
            return (pth, label, coef, r2)
        except Exception as e:
            print(f"[WARN] failed on #{idx} {pth}: {e}")
            return (pth, label, None, None)

    all_tasks = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    if args.progress_seq:
        results = []
        for i, (p, lab) in enumerate(tqdm(all_tasks, desc="NoiseProbe", unit="img")):
            results.append(_proc(p, lab, i))
    else:
        print(f"[INFO] running joblib with n_jobs={args.n_jobs} ...")
        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(_proc)(p, lab, i) for i, (p, lab) in enumerate(all_tasks)
        )

    # 过滤失败
    results = [r for r in results if r[2] is not None]
    if len(results) == 0:
        print("[ERR] no features extracted.")
        return

    # ----- 写 CSV 前：标签自检 & 按路径修复 -----
    import collections, re
    cnt = collections.Counter([lab for _, lab, _, _ in results])
    print(f"[CHECK] label counts (before fix): {cnt}")

    if len(cnt) < 2:
        print("[WARN] only one class in results; try path-based relabel...")
        fixed = []
        for path, label, coef, r2 in results:
            s = str(path).lower()
            if "/0_real/" in s or re.search(r"/0[_-]?real/", s): lab2 = 0
            elif "/1_fake/" in s or re.search(r"/1[_-]?fake/", s): lab2 = 1
            elif "fake" in s: lab2 = 1
            elif "real" in s: lab2 = 0
            else: lab2 = label
            fixed.append((path, lab2, coef, r2))
        results = fixed
        cnt = collections.Counter([lab for _, lab, _, _ in results])
        print(f"[CHECK] label counts (after path-fix): {cnt}")
        if len(cnt) < 2:
            print("[FATAL] still single-class after path-based relabel; abort to avoid bad CSV.")
            sys.exit(1)

    # 写 CSV（表头包含原指标 + 新指标）
    header = ["path", "label"]
    metric_names = []
    for n in noises:
        metric_names += [
            f"{n}_gradH", f"{n}_DoGret", f"{n}_DCTzrun", f"{n}_DCTmid",
            f"{n}_PSDring", f"{n}_PSDpeak",
            f"{n}_JPEGdmse_a0.2", f"{n}_JPEGdssim_a0.2", f"{n}_JPEGmid_a0.2",
            f"{n}_JPEGdmse_a0.6", f"{n}_JPEGdssim_a0.6", f"{n}_JPEGmid_a0.6",
            f"{n}_ResEnergy", f"{n}_ResRing"
        ]
    coef_cols = []
    for m in metric_names:
        coef_cols += [f"{m}_a", f"{m}_b", f"{m}_c"]
    header += coef_cols

    out_csv = args.out_csv
    Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for path, label, coef, r2 in results:
            w.writerow([path, int(label)] + [f"{v:.6g}" for v in coef.tolist()])
    print(f"[OK] features saved to {out_csv}")

def main_eval(args):
    # 从 CSV 评测（监督+无监督）
    import pandas as pd
    df = pd.read_csv(args.eval_csv)
    X = df.drop(columns=["path", "label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    # 监督（快速 baseline）
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    acc = accuracy_score(y, (prob >= 0.5).astype(int))
    print(f"[Supervised] AUC={auc:.4f}  ACC={acc:.4f}")
    # 无监督：仅用真实类拟合 OneClassSVM
    X_real = X[y == 0]
    if len(X_real) > 10:
        oc = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        oc.fit(X_real)
        score = -oc.decision_function(X)
        auc_u = roc_auc_score(y, score)
        th = np.median(score[y == 0])
        pred = (score >= th).astype(int)
        acc_u = accuracy_score(y, pred)
        print(f"[Unsupervised] AUC={auc_u:.4f}  ACC~(median-th)={acc_u:.4f}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, help="path to REAL images dir")
    ap.add_argument("--fake_dir", type=str, help="path to FAKE images dir")
    ap.add_argument("--out_csv", type=str, help="output CSV for features")
    ap.add_argument("--alphas", type=str, default="0,0.2,0.4,0.6,0.8")
    ap.add_argument("--noises", type=str, default="gaussian,poisson,quant")
    ap.add_argument("--max_images_per_class", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--max_size", type=int, default=512)
    ap.add_argument("--eval_csv", type=str, help="evaluate from CSV (supervised & unsupervised)")
    ap.add_argument("--no_dog", action="store_true", help="disable DoG keypoint metric for speed")
    ap.add_argument("--progress_seq", action="store_true", help="sequential processing with tqdm progress bar")
    args = ap.parse_args()

    if args.eval_csv and not (args.real_dir or args.fake_dir or args.out_csv):
        main_eval(args); sys.exit(0)

    if not (args.real_dir and args.fake_dir and args.out_csv):
        print("Usage: python noise_probe.py --real_dir ... --fake_dir ... --out_csv ... [--eval_csv ...]")
        sys.exit(1)

    main_extract(args)
