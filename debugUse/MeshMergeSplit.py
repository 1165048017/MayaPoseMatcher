# ------------------------------------------------------------
#  get_selected_meshes_numpy.py  (含重合顶点检测)
#  Maya 2022+  Python 3
# ------------------------------------------------------------
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import numpy as np
from typing import Tuple, List


# --------------------------------------------------
# 1. 抽取单个模型的顶点 / UV / vt
# --------------------------------------------------
def get_mesh_arrays_fast(transform_path: str):
    """
    返回:
        vertices : np.ndarray (N,3)  顶点世界坐标（顶点级）
        uvs      : np.ndarray (M,2)  UV 坐标（顶点级）
        vt       : np.ndarray (K,4)  face-vertex 映射
                     [face_id, vert_id, uv_id, vn_id]
    """
    sel = om2.MSelectionList()
    sel.add(transform_path)
    dag = sel.getDagPath(0)
    dag.extendToShape()
    if dag.apiType() != om2.MFn.kMesh:
        raise RuntimeError(f"{transform_path} 不是多边形网格。")

    mesh = om2.MFnMesh(dag)

    # 1) 顶点
    points = mesh.getPoints(om2.MSpace.kWorld)
    vertices = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)

    # 2) UV
    u_array, v_array = mesh.getUVs()
    uvs = np.column_stack((u_array, v_array)).astype(np.float64)

    # 3) 获取 face-vertex 级索引
    face_counts, vert_ids = mesh.getVertices()
    _, uv_ids   = mesh.getAssignedUVs()
    _, norm_ids = mesh.getNormalIds()

    # face_id 列
    face_ids = np.repeat(np.arange(len(face_counts), dtype=np.int32), face_counts)

    # 组装 4 列：face_id, vert_id, uv_id, vn_id
    vt = np.column_stack((face_ids, vert_ids, uv_ids, norm_ids)).astype(np.int32)

    return vertices, uvs, vt


# --------------------------------------------------
# 2. 获取所有选中模型
# --------------------------------------------------
def get_selected_meshes_numpy() -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    sel_list = om2.MGlobal.getActiveSelectionList()
    if sel_list.isEmpty():
        raise RuntimeError("请先至少选中一个多边形模型。")

    results = []
    for i in range(sel_list.length()):
        dag = sel_list.getDagPath(i)
        dag.extendToShape()
        if dag.apiType() != om2.MFn.kMesh:
            continue
        name = sel_list.getDagPath(i).fullPathName()
        v, u, vt = get_mesh_arrays_fast(name)
        results.append((name, v, u, vt))

    if not results:
        raise RuntimeError("选中的节点里没有任何多边形网格。")

    return results


# --------------------------------------------------
# 3. 重合顶点检测
# --------------------------------------------------
def find_overlaps(a: np.ndarray,
                  b: np.ndarray,
                  decimals: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
        coords : (L,3) 重合坐标
        idx_a  : (L,)  在 a 中的行索引
        idx_b  : (L,)  在 b 中的行索引
    """
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)

    a_r = np.round(a, decimals)
    b_r = np.round(b, decimals)

    a_view = a_r.view([('', a.dtype)] * 3)
    b_view = b_r.view([('', b.dtype)] * 3)

    _, idx_a, idx_b = np.intersect1d(
        a_view, b_view, assume_unique=False, return_indices=True
    )
    return a[idx_a], idx_a, idx_b


# --------------------------------------------------
# 4. Maya 场景内选中指定顶点
# --------------------------------------------------
def select_vertices(mesh_name: str, indices: np.ndarray):
    vtx_strings = [f"{mesh_name}.vtx[{i}]" for i in indices]
    if vtx_strings:
        cmds.select(vtx_strings, replace=True)
    else:
        cmds.select(clear=True)


# --------------------------------------------------
# 5. 一键检测两个选中模型的重合顶点
# --------------------------------------------------
def detect_and_select_overlaps(decimals: int = 3):
    """
    选中两个模型 → 检测重合顶点 → 在第一个模型上选中它们
    """
    

    _, idx1, idx2 = find_overlaps(v1, v2, decimals=decimals)

    select_vertices(name1, idx1)
    # select_vertices(name2, idx2)
    print(f"共找到 {len(idx1)} 个重合顶点，已在 {name1} 中选中。")
    return name1, name2, idx1, idx2

def findIdx(pos,posList):
    flag = 0
    for p in posList:
        if(pos==p):
            return flag
        flag = flag+1

def MergeMeshes(verts1,faces1,verts2,faces2,overlap_idx1,overlap_idx2):
     # 合并
    total_v = []
    total_f = []    
    for i in range(len(verts1)):
        total_v.append(verts1[i])
    for i in range(len(faces1)):
        total_f.append(faces1[i,:].tolist())

    total_map = np.zeros((verts2.shape[0],1))
    for bi in range(verts2.shape[0]):
        if(bi in overlap_idx2):
            total_map[bi] = overlap_idx1[findIdx(bi,overlap_idx2)]
        else:
            total_v.append(verts2[bi])
            total_map[bi] = len(total_v)-1

    # 处理面
    for fi in range(0,len(faces2)):
        tmp = []
        for vi in range(len(faces2[fi])):
            tmp.append(int(total_map[faces2[fi][vi]][0]))
        total_f.append(tmp)
    return total_v,total_map,total_f


# 将顶点和面写入obj
def writeWithColor(v,f,IdxsWithColor,name):
    fp = open(name,'w')
    for i in range(len(v)):
        if(len(IdxsWithColor)!=0):
            if i in IdxsWithColor:
                fp.write("v {0} {1} {2} 1 0 0\n".format(v[i][0],v[i][1],v[i][2]))
            else:
                fp.write("v {0} {1} {2} 1 1 1\n".format(v[i][0],v[i][1],v[i][2]))
        else:
            fp.write("v {0} {1} {2}\n".format(v[i][0],v[i][1],v[i][2]))

    for i in range(len(f)):
        for j in range(len(f[i])):
            if(j==0):
                fp.write("f {0} ".format(f[i][j]+1))
            else:
                fp.write("{0} ".format(f[i][j]+1))
            if(j==len(f[i])-1):
                fp.write("\n")       
    fp.close()
    
# --------------------------------------------------
# 6. 直接运行测试
# --------------------------------------------------
if __name__ == "__main__":
    try:
        meshes = get_selected_meshes_numpy()
        if len(meshes) != 2:
            raise RuntimeError("请只选中两个多边形模型。")

        name1, v1, uv1, vt1 = meshes[0]
        name2, v2, uv2, vt2 = meshes[1]
        
        detect_and_select_overlaps(decimals=3)
        print(vt1[0:4,...])
        # writeWithColor(fullbody_v,fullbody_f,[], outputPath)
    except RuntimeError as e:
        cmds.warning(str(e))