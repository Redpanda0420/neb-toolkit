#!/usr/bin/env python3
"""neb_endpoint_check.py — NEB 端点一致性检查

检查 NEB 包中「冻结端点」是否等于「对应端点的弛豫结构」。
用法:
    cd <解压后的 neb 包目录>        # 含 00/ 08/ 00_IS/ 08_FS/ POSCAR_0X
    conda run -n base python neb_endpoint_check.py [--cell 00/POSCAR]

输出:
    1) 端点 vs 弛豫前/后结构的 max|d| 与 rms (pymatgen StructureMatcher)
    2) 冒烟判据: 若 (端点 vs 弛豫前POSCAR) 的 rms ≈ (另一个端点 vs 其弛豫前) 的 rms
       → 提示端点可能是"对称性搬运"而非真实弛豫
    3) 迁移原子逐图像步进（验证插值连续性）
"""
import argparse
import os
import sys

import numpy as np


# ---------------- POSCAR/CONTCAR 读取 (不依赖 pymatgen) ----------------
def read_vasp(path):
    lines = open(path).read().splitlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    try:                                    # VASP5: 第6行是元素符号
        counts = [int(x) for x in lines[6].split()]
        start, dirline = 8, lines[7]
    except (ValueError, IndexError):        # VASP4: 第6行就是计数
        counts = [int(x) for x in lines[5].split()]
        start, dirline = 7, lines[6]
    n = sum(counts)
    coords = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(start, start + n)]
    )
    if not dirline.strip().lower().startswith("d"):
        coords = coords @ np.linalg.inv(lattice)   # cartesian -> fractional
    return lattice, np.array(counts), coords


def cart_disp(lattice, a, b):
    """min-image 分数差 -> 笛卡尔位移模长 (逐原子, 按索引对应)"""
    d = ((a - b + 0.5) % 1.0) - 0.5
    return np.linalg.norm(d @ lattice, axis=1)


def sm_compare(a, b):
    """可选: pymatgen StructureMatcher 判定对称等价 + rms"""
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.core import Structure
    except Exception as exc:                        # noqa: BLE001
        return None, f"(pymatgen 不可用: {exc})"
    sm = StructureMatcher(primitive_cell=False, scale=False)
    s1, s2 = Structure.from_file(a), Structure.from_file(b)
    fit = sm.fit(s1, s2)
    try:
        rms = sm.get_rms_dist(s1, s2)
    except Exception as exc:                        # noqa: BLE001
        rms = f"err:{exc}"
    return fit, rms


# ---------------- 主流程 ----------------
ENDPOINTS = [
    # (冻结端点, 弛豫后 CONTCAR, 弛豫前 POSCAR, 标签)
    ("00/POSCAR", "00_IS/CONTCAR", "00_IS/POSCAR", "IS (初始态)"),
    ("08/POSCAR", "08_FS/CONTCAR", "08_FS/POSCAR", "FS (终态)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="00/POSCAR",
                    help="用于测迁移原子步进的参考结构 (默认 00/POSCAR)")
    args = ap.parse_args()

    missing = [f for grp in ENDPOINTS for f in (grp[0],) if not os.path.exists(f)]
    if missing:
        sys.exit(f"❌ 找不到文件: {missing}  (请先 cd 到解压后的 neb 包目录)")

    print("=" * 78)
    print("1) 端点 vs 弛豫结构 一致性")
    print("=" * 78)
    print(f"{'对比':44s} {'fit':>5s} {'rms':>12s} {'max|d|':>9s} {'mean':>8s}")
    rms_vs_prerelax = {}
    for frozen, relax_con, relax_pre, label in ENDPOINTS:
        lat, _, x_frozen = read_vasp(frozen)
        for other, tag in ((relax_con, "弛豫后 CONTCAR"), (relax_pre, "弛豫前 POSCAR")):
            _, _, x_other = read_vasp(other)
            d = cart_disp(lat, x_frozen, x_other)
            fit, rms = sm_compare(frozen, other)
            rms_s = f"{rms[0]:.6f}" if isinstance(rms, tuple) else str(rms)
            print(f"{frozen+' vs '+other:44s} {str(fit):>5s} {rms_s:>12s} "
                  f"{d.max():9.4f} {d.mean():8.4f}   [{label} / {tag}]")
            if tag.startswith("弛豫前") and isinstance(rms, tuple):
                rms_vs_prerelax[frozen] = rms[0]

    print()
    print("=" * 78)
    print("2) 冒烟判据 (smoking gun)")
    print("=" * 78)
    if len(rms_vs_prerelax) == 2:
        a, b = rms_vs_prerelax["00/POSCAR"], rms_vs_prerelax["08/POSCAR"]
        print(f"  IS 端点 vs 其弛豫前 : rms = {a:.6f}")
        print(f"  FS 端点 vs 其弛豫前 : rms = {b:.6f}")
        if abs(a - b) < 1e-4:
            print("  ⚠️  两者几乎相同 → 提示其中一个端点是「对称性搬运」的位移场，")
            print("      而不是该端点自己的真实弛豫结果，请人工核对来源。")
        else:
            print("  ✅ 两个端点各自独立弛豫，未见搬运迹象。")

    print()
    print("=" * 78)
    print("3) 迁移原子逐图像步进 (插值连续性)")
    print("=" * 78)
    if os.path.exists(args.cell):
        lat, _, x0 = read_vasp(args.cell)
        imgs = sorted(f for f in os.listdir(".") if f.startswith("POSCAR_"))
        prev = None
        for f in imgs:
            _, _, x = read_vasp(f)
            li = x[5]                                   # 迁移 Li 索引 (本体系 = 5)
            d = ((li - x0[5] + 0.5) % 1.0) - 0.5
            tot = np.linalg.norm(d @ lat)
            step = "" if prev is None else f"  step={tot - prev:.4f} Å"
            print(f"  {f}: Li5=({li[0]:.4f},{li[1]:.4f},{li[2]:.4f})"
                  f"  |disp|={tot:.4f} Å{step}")
            prev = tot


if __name__ == "__main__":
    main()
