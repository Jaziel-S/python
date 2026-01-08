#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LS-DYNA rigid.new 转换为 STL 网格文件
- 解析 *NODE 和 *ELEMENT_SHELL
- 四边形拆成两个三角形
- 输出 ASCII STL
"""

import sys
import os
import re
import math
import tkinter as tk
from tkinter import filedialog

def parse_rigid_new(path):
    nodes = {}
    shells = []
    mode = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("$"):
                continue
            up = line.upper()
            if up.startswith("*NODE"):
                mode = "NODE"; continue
            elif up.startswith("*ELEMENT_SHELL") or up.startswith("*ELEMENT"):
                mode = "SHELL"; continue
            elif up.startswith("*"):
                mode = None; continue

            nums = re.findall(r"[-+]?\d+\.\d+E[-+]?\d+|[-+]?\d+", line)

            if mode == "NODE" and len(nums) >= 4:
                try:
                    nid = int(nums[0])
                    x, y, z = map(float, nums[1:4])
                    nodes[nid] = (x, y, z)
                except:
                    pass

            elif mode == "SHELL":
                try:
                    if len(nums) >= 6:
                        eid = int(nums[0])
                        n1, n2, n3, n4 = map(int, nums[2:6])
                        shells.append((eid, [n1, n2, n3, n4]))
                    elif len(nums) == 5:
                        eid = int(nums[0])
                        n1, n2, n3, n4 = map(int, nums[1:5])
                        shells.append((eid, [n1, n2, n3, n4]))
                    elif len(nums) == 4:
                        eid = int(nums[0])
                        n1, n2, n3 = map(int, nums[1:4])
                        shells.append((eid, [n1, n2, n3, n3]))
                except:
                    pass
    return nodes, shells

def facet_normal(p1, p2, p3):
    u = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
    v = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
    n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    length = math.sqrt(n[0]**2+n[1]**2+n[2]**2)
    if length == 0: return (0.0,0.0,0.0)
    return (n[0]/length, n[1]/length, n[2]/length)

def write_stl(out_path, nodes, shells):
    with open(out_path, "w") as f:
        f.write("solid rigid\n")
        for eid, conn in shells:
            pts = [nodes[nid] for nid in conn if nid in nodes]
            if len(pts) < 3: continue
            if pts[2] == pts[3]:  # 三角形
                p1,p2,p3 = pts[0],pts[1],pts[2]
                n = facet_normal(p1,p2,p3)
                f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {p1[0]} {p1[1]} {p1[2]}\n")
                f.write(f"      vertex {p2[0]} {p2[1]} {p2[2]}\n")
                f.write(f"      vertex {p3[0]} {p3[1]} {p3[2]}\n")
                f.write("    endloop\n  endfacet\n")
            else:  # 四边形拆成两个三角
                p1,p2,p3,p4 = pts
                for tri in [(p1,p2,p3),(p1,p3,p4)]:
                    n = facet_normal(*tri)
                    f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
                    f.write("    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
                    f.write("    endloop\n  endfacet\n")
        f.write("endsolid rigid\n")

def choose_file():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(title="请选择 rigid.new 文件",
        filetypes=[("LS-DYNA rigid.new","*.new"),("所有文件","*.*")])

def save_file(default_name="output.stl"):
    root = tk.Tk(); root.withdraw()
    return filedialog.asksaveasfilename(title="保存 STL 文件为...",
        defaultextension=".stl",initialfile=default_name,
        filetypes=[("STL 文件","*.stl"),("所有文件","*.*")])

def main():
    if len(sys.argv)<2:
        in_path = choose_file()
        if not in_path: print("未选择文件"); sys.exit(1)
        out_path = save_file(os.path.splitext(os.path.basename(in_path))[0]+".stl")
        if not out_path: print("未选择保存路径"); sys.exit(1)
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv)>2 else os.path.splitext(in_path)[0]+".stl"

    nodes,shells = parse_rigid_new(in_path)
    print(f"节点数: {len(nodes)}, 壳单元数: {len(shells)}")

    write_stl(out_path,nodes,shells)
    print(f"已导出 STL 文件: {out_path}")

if __name__=="__main__":
    main()
