# DYnaform5.9.4回弹补偿
回弹补偿会产生rigid.new模面的网格文件
---
🔧 思路
解析 rigid.new：读取 *NODE 和 *ELEMENT_SHELL。

节点表：保存节点号 → (x,y,z)。

单元表：保存四边形或三角形的节点号。

OBJ 输出：

v x y z → 节点坐标。

f n1 n2 n3 → 三角面。

f n1 n2 n3 n4 → 四边形面。

OBJ 格式天然支持四边形，所以能保留原始 FE 网格的密度。
