#!/usr/bin/env python3
"""path_vis.py — 把 NEB 迁移路径画到晶体结构上（输出可直接用 VESTA / OVITO / catgo 打开）

思路：
  1. 读所有 NEB 图像（每个子目录一个 POSCAR/CONTCAR），按索引一一对应
  2. 以端点位移最大者作为「迁移原子」
  3. 逐图像用**最近邻连续性**跟踪该原子（防索引错位）
  4. 相邻图像之间线性插值，得到约 --npts 个路径点
  5. 把这些点作为 ghost 原子（默认元素 Xe）追加到结构末尾，写出 VASP5 POSCAR
     可选同时写 CIF（需要 pymatgen）

用法:
    python path_vis.py images/                # images/00 images/01 ... images/08
    python path_vis.py images/ --npts 20 --ghost Xe --out path.poscar
    python path_vis.py images/ --cif          # 额外输出 path.cif (需 pymatgen)

参数:
    DIR                 图像根目录（子目录按名称排序，需含 POSCAR 或 CONTCAR）
    --pattern P         子目录匹配（默认 "*/"，即所有子目录）
    --npts N            路径插值点数（默认 20）
    --ghost ELEM        ghost 原子元素符号（默认 Xe）
    --out PATH          输出 POSCAR（默认 path.poscar）
    --cif               同时输出 CIF
    --tol Å             相邻图像同一原子的位移告警阈值（默认 3.0）

退出码: 0 正常；1 结构数/原子数不一致；2 未找到图像
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

POSCAR_NAMES = ("POSCAR", "CONTCAR", "POSCAR.vasp", "POSCAR.txt")


# ---------------- I/O ----------------
def read_vasp(path):
    """读 POSCAR/CONTCAR → (lattice, symbols, counts, frac)。支持 VASP4/5 + Selective dynamics。"""
    lines = open(path).read().splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale

    i = 5
    tokens = lines[i].split()
    if all(t.lstrip("+-").isdigit() for t in tokens):      # VASP4：无元素符号行
        symbols, counts = None, [int(t) for t in tokens]
        i += 1
    else:                                                    # VASP5
        symbols = tokens
        counts = [int(x) for x in lines[i + 1].split()]
        i += 2
    if lines[i].strip().lower().startswith("s"):             # Selective dynamics
        i += 1
    is_direct = lines[i].strip().lower().startswith("d")
    i += 1

    n = sum(counts)
    coords = np.array([[float(x) for x in lines[k].split()[:3]] for k in range(i, i + n)])
    if not is_direct:
        coords = coords @ np.linalg.inv(lattice)
    return lattice, symbols, counts, coords


def write_poscar(path, lattice, symbols, counts, frac, extra_symbols=()):
    """写 VASP5 POSCAR（直接坐标）。extra_symbols 追加到元素列表末尾。"""
    syms = list(symbols) + list(extra_symbols)
    cnts = list(counts) + [len(frac) - sum(counts)] if extra_symbols else list(counts)
    scale = 1.0
    with open(path, "w") as f:
        f.write("NEB migration path (ghost atoms appended)\n")
        f.write(f"  {scale:.8f}\n")
        for v in lattice:
            f.write("  " + "  ".join(f"{x:18.10f}" for x in v) + "\n")
        f.write("  " + "  ".join(syms) + "\n")
        f.write("  " + "  ".join(str(c) for c in cnts) + "\n")
        f.write("Direct\n")
        for c in frac:
            f.write("  " + "  ".join(f"{x:18.10f}" for x in c) + "\n")


def write_cif(path, lattice, symbols, counts, frac):
    try:
        from pymatgen.core import Lattice, Structure
    except Exception as exc:                        # noqa: BLE001
        print(f"[WARN] pymatgen 不可用，跳过 CIF：{exc}")
        return False
    species = []
    for s, c in zip(symbols, counts):
        species += [s] * c
    st = Structure(Lattice(lattice), species, frac, coords_are_cartesian=False)
    st.to(filename=path)
    return True


# ---------------- 几何 ----------------
def min_image_disp(lattice, a, b):
    """a、b 为分数坐标（可广播）→ 笛卡尔最短位移向量"""
    d = b - a
    d -= np.round(d)
    return d @ lattice


def find_migrating(lattice, c0, c1):
    """端点位移最大的原子索引"""
    d = np.linalg.norm(min_image_disp(lattice, c0, c1), axis=1)
    return int(np.argmax(d)), d


def track(lattice, frames, idx):
    """沿图像序列跟踪原子 idx：每步取**最小镜像位移**并累加，不依赖绝对坐标的包裹方式。

    做法：cur 为累计坐标（可落在周期盒外），每帧算 cur→全体原子的最小镜像位移，
    取最近者作为本步位移。这样跨周期边界时不会出现"跳一个晶格矢量"的假位移。
    """
    cur = frames[0][idx].astype(float).copy()
    path = [cur.copy()]
    for k in range(1, len(frames)):
        d = frames[k] - cur
        d -= np.round(d)                              # 最小镜像（分数坐标）
        dist = np.linalg.norm(d @ lattice, axis=1)    # 笛卡尔距离，仅用于选原子
        j = int(np.argmin(dist))
        cur = cur + d[j]
        path.append(cur.copy())
    return np.array(path)


def resample(path, npts):
    """按弧长等距重采样"""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(path[:1], npts, axis=0)
    targets = np.linspace(0.0, total, npts)
    out = []
    for t in targets:
        i = int(np.clip(np.searchsorted(s, t) - 1, 0, len(seg) - 1))
        w = (t - s[i]) / seg[i] if seg[i] > 0 else 0.0
        out.append(path[i] + w * (path[i + 1] - path[i]))
    return np.array(out)


# ---------------- main ----------------
def main():
    try:                                            # Windows 控制台默认 GBK，含 Å 会抛 UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                               # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="NEB 迁移路径可视化（ghost 原子链）")
    ap.add_argument("dir")
    ap.add_argument("--npts", type=int, default=20)
    ap.add_argument("--ghost", default="Xe")
    ap.add_argument("--out", default="path.poscar")
    ap.add_argument("--cif", action="store_true")
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()

    subdirs = sorted(d for d in glob.glob(os.path.join(args.dir, "*")) if os.path.isdir(d))
    frames, lattice, symbols, counts = [], None, None, None
    # 默认只取纯数字命名的图像目录（00/01/...），排除 00_IS / 08_FS 这类端点副本
    numbered = [(int(os.path.basename(d)), d) for d in subdirs
                if re.fullmatch(r"\d+", os.path.basename(d))]
    if len(numbered) >= 2:
        subdirs = [d for _, d in sorted(numbered)]
        skipped = [os.path.basename(d) for d in glob.glob(os.path.join(args.dir, "*"))
                   if os.path.isdir(d) and not re.fullmatch(r"\d+", os.path.basename(d))]
        if skipped:
            print(f"[INFO] 已按数字排序取 {len(subdirs)} 个图像；跳过非数字目录: {', '.join(sorted(skipped))}")
    for d in subdirs:
        p = next((os.path.join(d, n) for n in POSCAR_NAMES if os.path.exists(os.path.join(d, n))), None)
        if p is None:
            continue
        lat, sym, cnt, frac = read_vasp(p)
        if lattice is None:
            lattice, symbols, counts = lat, sym, cnt
        elif sum(cnt) != sum(counts):
            print(f"[ERROR] 原子数不一致: {d} ({sum(cnt)}) vs 首帧 ({sum(counts)})")
            return 1
        frames.append(frac)
    if len(frames) < 2:
        print(f"[ERROR] 找到的图像不足 2 个（{len(frames)}）: {args.dir}")
        return 2

    idx, d = find_migrating(lattice, frames[0], frames[-1])
    if symbols is None:
        print("[ERROR] VASP4 格式（无元素符号行）不支持输出，请补上第 6 行元素符号")
        return 1
    print(f"[OK] 图像 {len(frames)} 个；迁移原子索引 {idx}，端点位移 {d[idx]:.3f} Å")
    order = np.argsort(d)[::-1][:5]
    print("     位移 Top5: " + ", ".join(f"#{i}:{d[i]:.2f}Å" for i in order))
    if d[idx] < 0.5:
        print("[WARN] 最大位移 < 0.5 Å，可能端点选错了（迁移原子不明显）")

    tracked = track(lattice, frames, idx)
    steps = np.linalg.norm(np.diff(tracked, axis=0) @ lattice, axis=1)
    if steps.max() > args.tol:
        print(f"[WARN] 跟踪步长 {steps.max():.2f} Å 超过 --tol {args.tol}，查一下是否跟踪到了错原子")
    print("     逐图像步进: " + " ".join(f"{x:.2f}" for x in steps))

    path = resample(tracked, args.npts)
    allfrac = np.vstack([frames[0], path])
    write_poscar(args.out, lattice, symbols, counts, allfrac, extra_symbols=(args.ghost,))
    print(f"[OK] 写出 {args.out}（{args.npts} 个 {args.ghost} ghost 原子）")

    if args.cif:
        cif = os.path.splitext(args.out)[0] + ".cif"
        if write_cif(cif, lattice, list(symbols) + [args.ghost],
                     list(counts) + [args.npts], allfrac):
            print(f"[OK] 写出 {cif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
