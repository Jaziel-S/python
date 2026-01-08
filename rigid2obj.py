#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LS-DYNA rigid.new 转换为 OBJ 网格文件
- 保留四边形单元，不拆成三角
- 输出 ASCII OBJ
"""

import sys
import os
import re
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
                        shells.append((eid, [n1, n2, n3, n3]))  # 三角补齐
                except:
                    pass
    return nodes, shells

def write_obj(out_path, nodes, shells):
    # 节点号映射到 OBJ 索引（从1开始）
    node_index = {}
    vertices = []
    for i, (nid, coord) in enumerate(nodes.items(), start=1):
        node_index[nid] = i
        vertices.append(coord)

    with open(out_path, "w") as f:
        f.write("# OBJ file generated from rigid.new\n")
        # 写顶点
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # 写面
        for eid, conn in shells:
            idxs = [node_index[nid] for nid in conn if nid in node_index]
            if len(idxs) == 3 or (len(idxs) == 4 and idxs[2] == idxs[3]):
                f.write(f"f {idxs[0]} {idxs[1]} {idxs[2]}\n")
            elif len(idxs) == 4:
                f.write(f"f {idxs[0]} {idxs[1]} {idxs[2]} {idxs[3]}\n")

def choose_file():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(title="请选择 rigid.new 文件",
        filetypes=[("LS-DYNA rigid.new","*.new"),("所有文件","*.*")])

def save_file(default_name="output.obj"):
    root = tk.Tk(); root.withdraw()
    return filedialog.asksaveasfilename(title="保存 OBJ 文件为...",
        defaultextension=".obj",initialfile=default_name,
        filetypes=[("OBJ 文件","*.obj"),("所有文件","*.*")])

def main():
    if len(sys.argv)<2:
        in_path = choose_file()
        if not in_path: print("未选择文件"); sys.exit(1)
        out_path = save_file(os.path.splitext(os.path.basename(in_path))[0]+".obj")
        if not out_path: print("未选择保存路径"); sys.exit(1)
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv)>2 else os.path.splitext(in_path)[0]+".obj"

    nodes,shells = parse_rigid_new(in_path)
    print(f"节点数: {len(nodes)}, 壳单元数: {len(shells)}")

    write_obj(out_path,nodes,shells)
    print(f"已导出 OBJ 文件: {out_path}")

if __name__=="__main__":
    main()
