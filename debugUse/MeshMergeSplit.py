import maya.cmds as cmds

def merge_models_by_uv_udim():
    selection = cmds.ls(selection=True)
    if len(selection) != 2:
        cmds.warning("请选中两个模型")
        return

    model_a, model_b = selection

    cmds.inViewMessage(amg="📌 正在移动 UV 到 UDIM…", pos='topCenter', fade=True)
    cmds.select(model_b + ".map[*]", r=True)
    cmds.polyEditUV(u=1.0, v=0.0)

    cmds.inViewMessage(amg="📌 正在合并模型…", pos='topCenter', fade=True)
    merged = cmds.polyUnite(model_a, model_b, ch=False, mergeUVSets=True, name="merged_model")[0]
    cmds.delete(merged, ch=True)

    cmds.inViewMessage(amg="⏳ 正在合并重合顶点（可能较慢）…", pos='topCenter', fade=True)
    cmds.select(merged)
    cmds.polyMergeVertex(d=0.0001)

    cmds.select(merged)
    cmds.inViewMessage(amg="✅ 合并完成，UV 已分区", pos='topCenter', fade=True)

def split_model_by_uv_udim():
    sel = cmds.ls(selection=True)
    if not sel:
        cmds.warning("请选中合并后的模型")
        return    
    merged = sel[0]
    vtx_count = cmds.polyEvaluate(merged, vertex=True)

    # 获取每个顶点的 UV 坐标
    vtx_to_uv = {}
    for i in range(vtx_count):
        uv = cmds.polyListComponentConversion(f"{merged}.vtx[{i}]", fromVertex=True, toUV=True)
        uv = cmds.filterExpand(uv, selectionMask=35)
        if uv:
            uv_pos = cmds.polyEditUV(uv[0], query=True)
            vtx_to_uv[i] = uv_pos

    # 根据 UV 的 U 值分类
    group_a = [i for i, uv in vtx_to_uv.items() if uv[0] < 1.0]
    group_b = [i for i, uv in vtx_to_uv.items() if uv[0] >= 1.0]

    def create_mesh_from_vertices(vtx_indices, name):
        positions = [cmds.pointPosition(f"{merged}.vtx[{i}]", world=True) for i in vtx_indices]
        mesh = cmds.polyCreateFacet(p=positions, name=name)[0]
        return mesh    
    mesh_a = create_mesh_from_vertices(group_a, "reconstructed_model_A")
    mesh_b = create_mesh_from_vertices(group_b, "reconstructed_model_B")

    cmds.select(mesh_a, mesh_b)
    cmds.inViewMessage(amg="🔄 拆分完成，恢复为两个模型", pos='topCenter', fade=True)

def show_uv_udim_merge_split_ui():
    if cmds.window("uvUdimMergeSplitUI", exists=True):
        cmds.deleteUI("uvUdimMergeSplitUI")

    window = cmds.window("uvUdimMergeSplitUI", title="UV UDIM 合并拆分工具", widthHeight=(300, 120))
    cmds.columnLayout(adjustableColumn=True, rowSpacing=10, columnAlign="center")

    cmds.text(label="请选择两个模型进行合并，或选择合并模型进行拆分")
    cmds.button(label="🔗 合并模型（UV 分区）", command=lambda x: merge_models_by_uv_udim())
    cmds.button(label="🔄 拆分模型（基于 UV）", command=lambda x: split_model_by_uv_udim())

    cmds.setParent("..")
    cmds.showWindow(window)

# 启动 UI
show_uv_udim_merge_split_ui()
