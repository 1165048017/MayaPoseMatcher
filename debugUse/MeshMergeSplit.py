# ------------------------------------------------------------
#  get_selected_meshes_numpy.py  (含重合顶点检测)
#  Maya 2022+  Python 3
# ------------------------------------------------------------
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import numpy as np
import math
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

    # 3) VN
    normals = np.array(mesh.getNormals())

    # 3) 获取 face-vertex 级索引
    face_counts, vert_ids = mesh.getVertices()
    _, uv_ids   = mesh.getAssignedUVs()
    _, norm_ids = mesh.getNormalIds()

    # face_id 列
    face_ids = np.repeat(np.arange(len(face_counts), dtype=np.int32), face_counts)

    # 组装 4 列：face_id, vert_id, uv_id, vn_id
    vt = np.column_stack((face_ids, vert_ids, uv_ids, norm_ids)).astype(np.int32)

    return vertices, uvs, normals, vt


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
        v, u, vn, vt = get_mesh_arrays_fast(name)
        results.append((name, v, u, vn, vt))

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
    print(f"共找到 {len(idx1)} 个重合顶点，已在 {name1} 中选中。")
    return name1, name2, idx1, idx2

# --------------------------------------------------
# 5. 合并模型
# --------------------------------------------------
# 将顶点和面写入obj
def writeWithColor(f,v,uv,vn,IdxsWithColor,name):
    fp = open(name,'w')
    for i in range(v.shape[0]):
        if(IdxsWithColor!=[] and IdxsWithColor.shape[0]!=0):
            if i in IdxsWithColor:
                fp.write("v {0} {1} {2} 1 0 0\n".format(v[i,0],v[i,1],v[i,2]))
            else:
                fp.write("v {0} {1} {2} 1 1 1\n".format(v[i,0],v[i,1],v[i,2]))
        else:
            fp.write("v {0} {1} {2}\n".format(v[i,0],v[i,1],v[i,2]))
    for i in range(uv.shape[0]):        
        fp.write("uv {0} {1}\n".format(uv[i,0],uv[i,1]))
    for i in range(vn.shape[0]):        
        fp.write("vn {0} {1} {2}\n".format(vn[i,0],vn[i,1],vn[i,2]))
    curr_fid = -1
    for i in range(f.shape[0]):
        if(f[i,0]==curr_fid):
            fp.write(" {0}/{1}/{2}".format(f[i,1]+1,f[i,2]+1,f[i,3]+1))
        else:
            fp.write("\nf {0}/{1}/{2}".format(f[i,1]+1,f[i,2]+1,f[i,3]+1))
            curr_fid = f[i,0]
    # for i in range(len(f)):
    #     for j in range(len(f[i])):
    #         if(j==0):
    #             fp.write("f {0} ".format(f[i][j]+1))
    #         else:
    #             fp.write("{0} ".format(f[i][j]+1))
    #         if(j==len(f[i])-1):
    #             fp.write("\n")       
    fp.close()

def ProcessVertices(mesh1_v,mesh2_v,overlap1,overlap2):
    total_v = mesh1_v.copy()
    map_v = np.array([], dtype=np.int32)
    for i in range(mesh2_v.shape[0]):
        if(i not in overlap2):
            total_v = np.vstack([total_v,mesh2_v[i,...]])
            map_v = np.hstack([map_v, np.array([total_v.shape[0]-1], dtype=np.int32)])
        else:
            indices = np.where(i == overlap2)[0]
            map_v = np.hstack([map_v,overlap1[indices[0]]])
    return total_v, map_v

def is_overlap(a, b):
    """
    a, b: (x_min, y_min, x_max, y_max)
    边贴合不算重叠
    返回 True 表示两矩形严格重叠
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 任一方向不重叠 → 整体不重叠
    no_overlap_x = ax2 <= bx1 or bx2 <= ax1
    no_overlap_y = ay2 <= by1 or by2 <= ay1

    # 反推：两个方向都不“不重叠”才重叠
    return not (no_overlap_x or no_overlap_y)

def ProcessUV(mesh1_uv,mesh2_uv):
    merged_uv = mesh1_uv.copy()
    uv1_range = [math.floor(min(mesh1_uv[...,0])), math.floor(min(mesh1_uv[...,1])),math.ceil(max(mesh1_uv[...,0])), math.ceil(max(mesh1_uv[...,1]))]
    uv2_range = [math.floor(min(mesh2_uv[...,0])), math.floor(min(mesh2_uv[...,1])),math.ceil(max(mesh2_uv[...,0])), math.ceil(max(mesh2_uv[...,1]))]
    if(is_overlap(uv1_range,uv2_range)):
        if(uv1_range[0]<uv2_range[0]):
            mesh2_uv[...,0] += uv1_range[3]-uv2_range[0]
    merged_uv = np.vstack([mesh1_uv,mesh2_uv])
    map_uv = np.arange(mesh2_uv.shape[0]) + mesh1_uv.shape[0]
    return merged_uv, map_uv

def ProcessVN(mesh1_vn,mesh2_vn):
    return np.vstack([mesh1_vn,mesh2_vn]), np.arange(mesh2_vn.shape[0])+mesh1_vn.shape[0]

def ProcessFaces(mesh1_f,mesh2_f,map_v,map_uv,map_vn):
    cmds.progressWindow(
        title='Processing faces...',
        progress=0,
        status='0 / %d' % mesh2_f.shape[0],
        isInterruptable=True
    )

    merged_f = mesh1_f.copy()
    for i in range(mesh2_f.shape[0]):
        # ---- 每帧检查是否用户点了取消 ----
        if cmds.progressWindow(query=True, isCancelled=True):
            cmds.warning('用户取消！')
            break

        cur_face_id = mesh2_f[i,0] + mesh1_f.shape[0]
        cur_vert_id = map_v[mesh2_f[i,1]]
        cur_uv_id = map_uv[mesh2_f[i,2]]
        cur_norm_id = map_vn[mesh2_f[i,3]]
        merged_f = np.vstack([merged_f,np.column_stack((cur_face_id, cur_vert_id, cur_uv_id, cur_norm_id)).astype(np.int32)])

        # ---- 更新进度 ----
        percent = float(i + 1) / mesh2_f.shape[0] * 100
        cmds.progressWindow(edit=True,
                          progress=percent,
                          status='%d / %d' % (i + 1, mesh2_f.shape[0]))
    cmds.progressWindow(endProgress=True)
    return merged_f


def MergeMeshes(mesh1_f,mesh2_f,mesh1_v,mesh2_v,mesh1_uv,mesh2_uv,mesh1_vn,mesh2_vn,overlap1,overlap2):
    merged_v, map_v = ProcessVertices(mesh1_v,mesh2_v,overlap1,overlap2)
    merged_uv, map_uv = ProcessUV(mesh1_uv,mesh2_uv)
    merged_vn, map_vn = ProcessVN(mesh1_vn,mesh2_vn)
    merged_f = ProcessFaces(mesh1_f,mesh2_f,map_v,map_uv,map_vn)
    return merged_f, merged_v,merged_uv,merged_vn

# --------------------------------------------------
# 6. 创建mesh
# --------------------------------------------------
def build_mesh_from_numpy(v, uv, vt, vn, name='newMesh'):
    """
    v   : (N,3)  float  顶点
    uv  : (M,2)  float  顶点级 UV
    vt  : (K,4)  int    face-vertex 映射 [face_id, vert_id, uv_id, vn_id]
    vn  : (P,3)  float  顶点级法线
    """
    # 1. 基本检查
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError('v 必须是 (N,3)')
    if vt.shape[1] != 4:
        raise ValueError('vt 必须是 (K,4)')

    # 2. 构造 MPointArray
    points = om2.MPointArray([om2.MPoint(*p) for p in v])

    # 3. 构造 faceCounts 与 faceConnects
    # vt 已按 face_id 连续排列，每 3 个一组就是三角形
    face_counts  = [3] * (vt.shape[0] // 3)
    face_connects = om2.MIntArray(vt[:, 1].tolist())  # vert_id 列

    # 4. 创建空 mesh
    mfn = om2.MFnMesh()
    mesh_obj = mfn.create(points, face_counts, face_connects, name=name)

    # 5. 写入 UV
    u_arr = om2.MFloatArray(uv[:, 0].tolist())
    v_arr = om2.MFloatArray(uv[:, 1].tolist())
    mfn.setUVs(u_arr, v_arr)
    uv_ids = om2.MIntArray(vt[:, 2].tolist())   # uv_id 列
    mfn.assignUVs(om2.MIntArray(face_counts), uv_ids)

    # 6. 写入法线（顶点级）
    norm_arr = om2.MVectorArray([om2.MVector(*n) for n in vn])
    norm_ids = om2.MIntArray(vt[:, 3].tolist())  # vn_id 列
    mfn.setVertexNormals(norm_arr, norm_ids)

    # 7. 改名
    new_tf = om2.MFnTransform(mesh_obj).name()   # 拿到 transform 节点
    cmds.rename(new_tf, name)
    cmds.select(new_tf, r=True)
    return new_tf
# --------------------------------------------------
# 7. 直接运行测试
# --------------------------------------------------
if __name__ == "__main__":
    try:
        meshes = get_selected_meshes_numpy()
        if len(meshes) != 2:
            raise RuntimeError("请只选中两个多边形模型。")

        name1, v1, uv1,vn1, vt1 = meshes[0]
        name2, v2, uv2,vn2, vt2 = meshes[1]
        
        name1, name2, overlap_idx1, overlap_idx2 = detect_and_select_overlaps(decimals=3)
        select_vertices(name1, overlap_idx1)
        # select_vertices(name2, idx2)
        merged_f, merged_v,merged_uv,merged_vn = MergeMeshes(vt1,vt2,v1,v2,uv1,uv2,vn1,vn2,overlap_idx1,overlap_idx2)
        
        # np.save('E:/Code/python/MayaPoseMatcher/debugUse/merged_f.npy',merged_f)
        # np.save('E:/Code/python/MayaPoseMatcher/debugUse/merged_v.npy',merged_v)
        # np.save('E:/Code/python/MayaPoseMatcher/debugUse/merged_uv.npy',merged_uv)
        # np.save('E:/Code/python/MayaPoseMatcher/debugUse/merged_vn.npy',merged_vn)
        # np.save('E:/Code/python/MayaPoseMatcher/debugUse/overlap_idx1.npy',overlap_idx1)

        # merged_f = np.load('E:/Code/python/MayaPoseMatcher/debugUse/merged_f.npy')
        # merged_v = np.load('E:/Code/python/MayaPoseMatcher/debugUse/merged_v.npy')
        # merged_uv = np.load('E:/Code/python/MayaPoseMatcher/debugUse/merged_uv.npy')
        # merged_vn = np.load('E:/Code/python/MayaPoseMatcher/debugUse/merged_vn.npy')
        # overlap_idx1 = np.load('E:/Code/python/MayaPoseMatcher/debugUse/overlap_idx1.npy')

        build_mesh_from_numpy(merged_v,merged_uv,merged_f, merged_vn, name='MyMergedMesh')

        # writeWithColor(merged_f,merged_v,merged_uv,merged_vn, overlap_idx1, "E:/Code/python/MayaPoseMatcher/debugUse/merged.obj")
    except RuntimeError as e:
        cmds.warning(str(e))