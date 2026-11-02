#!/usr/bin/env python3
"""plot_barrier.py — 从 NEB 图像目录提取能量并出势垒曲线

支持两种数据源：
  1. 图像目录（默认）：<dir>/00/OUTCAR, <dir>/01/OUTCAR ... 取每个 OUTCAR 末次 free energy
  2. VTST 的 .dat / .ener 文本：--ener 文件，逐行 "距离 能量"

输出：
  - 表格打印（相对首像的 ΔE，eV 与 meV）
  - <out>.csv
  - <out>.png（有 matplotlib 时）

用法:
    python plot_barrier.py images/ --out barrier
    python plot_barrier.py --ener path.dat --out barrier --xlabel "Reaction coordinate (Å)"
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np


def final_free_energy(outcar):
    """取最后一个 'free  energy   TOTEN' 行（eV）"""
    e = None
    with open(outcar, "r", errors="ignore") as f:
        for line in f:
            if "free  energy" in line and "TOTEN" in line:
                m = re.search(r"=\s*(-?\d+\.\d+)", line)
                if m:
                    e = float(m.group(1))
    return e


def read_ener(path):
    """读 "x E" 两列表（跳过非数值行）"""
    xs, es = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                es.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(xs), np.array(es)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                               # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="NEB 势垒曲线")
    ap.add_argument("dir", nargs="?", help="图像根目录（含 00/ 01/ ... 子目录）")
    ap.add_argument("--ener", help="直接给 VTST .dat/.ener 文件")
    ap.add_argument("--out", default="barrier")
    ap.add_argument("--xlabel", default="Image")
    ap.add_argument("--title", default="NEB energy profile")
    args = ap.parse_args()

    if args.ener:
        x, e = read_ener(args.ener)
        if len(e) == 0:
            print(f"[ERROR] {args.ener} 里没有可解析的数据行")
            return 1
    elif args.dir:
        dirs = sorted(
            (d for d in glob.glob(os.path.join(args.dir, "*"))
             if os.path.isdir(d) and re.fullmatch(r"\d+", os.path.basename(d))),
            key=lambda d: int(os.path.basename(d)),
        )
        e = []
        for d in dirs:
            p = os.path.join(d, "OUTCAR")
            if not os.path.exists(p):
                print(f"[WARN] 跳过无 OUTCAR 的 {d}")
                continue
            v = final_free_energy(p)
            if v is None:
                print(f"[WARN] {p} 里没找到 free energy")
                continue
            e.append(v)
        e = np.array(e)
        x = np.arange(len(e), dtype=float)
        if len(e) < 2:
            print("[ERROR] 有效图像不足 2 个")
            return 1
        print(f"[INFO] 读取 {len(e)} 个图像")
    else:
        ap.error("需要 DIR 或 --ener")

    d_e = (e - e[0]) * 1000.0                       # meV，相对首像
    imax = int(np.argmax(e))
    print("\n  image   E (eV)        ΔE (meV)")
    for i, (ev, dv) in enumerate(zip(e, d_e)):
        flag = "  <== TS" if i == imax else ""
        print(f"  {i:>5}   {ev:.6f}   {dv:>10.1f}{flag}")
    print(f"\n[OK] 势垒 Ea = {d_e[imax]:.1f} meV = {d_e[imax]/1000:.4f} eV （最高点 @ image {imax}）")
    if len(e) > 2:
        rev = d_e[-1] - d_e[imax]
        print(f"     反向势垒 = {rev:.1f} meV；端点能量差 ΔE = {d_e[-1]:.1f} meV")
        if abs(d_e[-1]) > 100:
            print("[WARN] 端点能量差 > 100 meV：两端点很可能不是同一套参数弛豫的，势垒不可比")

    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "x", "E_eV", "dE_meV"])
        for i in range(len(e)):
            w.writerow([i, x[i], f"{e[i]:.6f}", f"{d_e[i]:.1f}"])
    print(f"[OK] 写出 {args.out}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                        # noqa: BLE001
        print(f"[INFO] 无 matplotlib，跳过绘图（{exc}）")
        return 0
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    ax.plot(x, d_e, "o-", color="#1f3b73", lw=1.6, ms=4)
    ax.plot(x[imax], d_e[imax], "o", color="#c00000", ms=7, zorder=3)
    ax.annotate(f"Ea = {d_e[imax]:.0f} meV", (x[imax], d_e[imax]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, color="#c00000")
    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(r"$\Delta E$ (meV)")
    ax.set_title(args.title, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out + ".png")
    print(f"[OK] 写出 {args.out}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
