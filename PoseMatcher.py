import maya.cmds as cmds
import maya.api.OpenMaya as om
import numpy as np
import math
import json
import os
import re

def get_base_name(obj_name):
    """获取对象的基名称（去除命名空间和路径）"""
    return obj_name.split(":")[-1].split("|")[-1]

def apply_global_rotation_to_vector(rotation_quaternion, vector):
    # 4. 创建一个 MVector 对象来表示您要旋转的向量
    vector_mvector = om.MVector(vector[0], vector[1], vector[2])
    # 6. 使用 rotateBy 方法将旋转应用到向量上
    rotated_vector = vector_mvector.rotateBy(rotation_quaternion)
    return [rotated_vector.x, rotated_vector.y, rotated_vector.z]

def getlocalRotation(joint_name):
    return cmds.xform(joint_name, query=True, worldSpace=False, rotation=True)

def calculate_angle_3d(a, b):
    a = np.array(a)
    b = np.array(b)    
    dot_product = np.dot(a, b)    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    theta = np.arccos(cos_theta)    
    return np.degrees(theta)

def get_joint_lra(joint, length=10.0):
    # 获取关节世界位置
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)

    # 获取 jointOrient 并转换为旋转矩阵
    orient = cmds.getAttr(joint + ".jointOrient")[0]
    euler = om.MEulerRotation(
        math.radians(orient[0]),
        math.radians(orient[1]),
        math.radians(orient[2])
    )
    rot_matrix = euler.asMatrix()

    # 提取旋转轴方向向量
    x_axis = om.MVector(rot_matrix[0], rot_matrix[1], rot_matrix[2]).normal()
    y_axis = om.MVector(rot_matrix[4], rot_matrix[5], rot_matrix[6]).normal()
    z_axis = om.MVector(rot_matrix[8], rot_matrix[9], rot_matrix[10]).normal()
    # 计算终点
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length
    xyzVec = []
    xyzVec.append([x_end.x-origin.x, x_end.y-origin.y, x_end.z-origin.z])
    xyzVec.append([y_end.x-origin.x, y_end.y-origin.y, y_end.z-origin.z])
    xyzVec.append([z_end.x-origin.x, z_end.y-origin.y, z_end.z-origin.z])
    return xyzVec

def get_joint_global_lra(joint, length=100.0):
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)

    matrix = cmds.xform(joint, q=True, m=True, ws=True)
    m_matrix = om.MMatrix(matrix)

    joint_axis = cmds.getAttr(joint + ".rotateAxis")[0]
    order = cmds.getAttr(joint + ".rotateOrder")
    euler = om.MEulerRotation(
        math.radians(joint_axis[0]),
        math.radians(joint_axis[1]),
        math.radians(joint_axis[2]),
        order
    )
    euler_matrix = euler.asMatrix()
    rot_matrix = m_matrix*(euler_matrix.inverse())
    # 提取旋转轴方向向量
    x_axis = om.MVector(rot_matrix[0], rot_matrix[1], rot_matrix[2]).normal()
    y_axis = om.MVector(rot_matrix[4], rot_matrix[5], rot_matrix[6]).normal()
    z_axis = om.MVector(rot_matrix[8], rot_matrix[9], rot_matrix[10]).normal()
    # 计算终点
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length
    xyzVec = []
    xyzVec.append([x_end.x-origin.x, x_end.y-origin.y, x_end.z-origin.z])
    xyzVec.append([y_end.x-origin.x, y_end.y-origin.y, y_end.z-origin.z])
    xyzVec.append([z_end.x-origin.x, z_end.y-origin.y, z_end.z-origin.z])
    return xyzVec

def findClose0or180(angles):
    newAngles= []
    for angle in angles:
        print(angle)
        newAngles.append(min(angle,180-abs(angle)))
    print(newAngles)
    min_value = min(newAngles)  # 找到最小值
    min_index = newAngles.index(min_value)  # 找到最小值的索引
    print(min_index)
    return min_index

def getAimUpVector(joint_name, boneDir):
    lraVec = get_joint_global_lra(joint_name)
    angle_x = calculate_angle_3d(np.array(lraVec[0]),np.array(boneDir))
    angle_y = calculate_angle_3d(np.array(lraVec[1]),np.array(boneDir))
    angle_z = calculate_angle_3d(np.array(lraVec[2]),np.array(boneDir))
    axis = [[1,0,0],[0,1,0],[0,0,1]]
    angles = [angle_x, angle_y, angle_z]
    print("angles:")
    print(angles)
    min_index = findClose0or180(angles)
    sign = 1
    if(angles[min_index]>180-abs(angles[min_index])):
        sign = -1
    
    aimVector = [sign*axis[min_index][0],sign*axis[min_index][1],sign*axis[min_index][2]]
    print(f"{joint_name} aim vector:")
    print(aimVector)
    return [aimVector, axis[(min_index+1)%3]]

def rotate_joint_to_direction(joint_name,aimVector,target_direction):
    # 创建一个空对象作为目标
    target_object = cmds.spaceLocator(name="target")[0]
    # 设置空对象的位置为关节1的位置，以确保它与关节1重合
    cmds.delete(cmds.parentConstraint(joint_name, target_object))
    # 将空对象的位置移动到目标方向
    global_position = cmds.xform(joint_name, query=True, worldSpace=True, translation=True)
    cmds.move(global_position[0]+target_direction[0], global_position[1]+target_direction[1], global_position[2]+target_direction[2], target_object, absolute=True)
    # 使用 aimConstraint 使关节2指向空对象
    cmds.aimConstraint(target_object, joint_name, aimVector=aimVector)
    # if("_l" in joint_name or "l_" in joint_name):
    #     cmds.aimConstraint(target_object, joint_name, aimVector=[1,0,0],upVector=[0,1,0])
    # else:
    #     cmds.aimConstraint(target_object, joint_name, aimVector=[-1,0,0],upVector=[0,1,0])
    # 删除临时创建的目标对象
    # cmds.delete(target_object)

def normalize_vector(v):
    """将向量归一化"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_joint_position(joint):
    """获取关节的世界坐标位置"""
    pos = cmds.xform(joint, query=True, worldSpace=True, translation=True)
    return np.array(pos)

def getGlobalRot(joint):
    matrix = cmds.xform(joint, q=True, m=True, ws=True)
    m_matrix = om.MMatrix(matrix)
    t_matrix = om.MTransformationMatrix(m_matrix)
    return t_matrix.rotation(asQuaternion=True).asMatrix()

def getJointOrient(joint):
    orient = cmds.getAttr(joint + ".jointOrient")[0]
    jo_euler = om.MEulerRotation(
        math.radians(orient[0]),
        math.radians(orient[1]),
        math.radians(orient[2])
    )
    jo_matrix = jo_euler.asMatrix()
    return jo_matrix

def getRotateAxisRot(joint):
    joint_axis = cmds.getAttr(joint + ".rotateAxis")[0]
    order = cmds.getAttr(joint + ".rotateOrder")
    ra_euler = om.MEulerRotation(
        math.radians(joint_axis[0]),
        math.radians(joint_axis[1]),
        math.radians(joint_axis[2]),
        0
    )
    ra_matrix = ra_euler.asMatrix()
    return ra_matrix

def getRotOrder(joint):
    order = cmds.getAttr(joint + ".rotateOrder")
    return order

def getAnimRotate(joint):
    animLocalRot = cmds.getAttr(joint + ".rotate")[0]
    animEuler = om.MEulerRotation(
        math.radians(animLocalRot[0]),
        math.radians(animLocalRot[1]),
        math.radians(animLocalRot[2]),
        getRotOrder(joint)
    )
    anim_matrix = animEuler.asMatrix()
    return anim_matrix

def getInherentRotWithoutParent(joint):
    jo_matrix = getJointOrient(joint)
    ra_matrix = getRotateAxisRot(joint)
    anim_matrix = getAnimRotate(joint)

    tmp_matrix=om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    result_matrix=om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp_matrix.setToProduct(ra_matrix,anim_matrix)
    result_matrix.setToProduct(tmp_matrix,jo_matrix)
    return result_matrix

def get_parent_delta_global_rot(joint_name):
    inherent_global_rot = getInherentRotWithoutParent(joint_name)
    global_rot = getGlobalRot(joint_name)
    delta_rot = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    delta_rot.setToProduct(inherent_global_rot.inverse(), global_rot)
    return delta_rot

def getNeedRot(v1:om.MVector,v2:om.MVector):
    rot_quat = v1.rotateTo(v2)
    return rot_quat.asMatrix()

def align_parent_to_vector(child_joint, target_vector):
    """将父关节指向子关节的方向对齐到目标向量"""
    # 获取父关节和子关节的位置
    parent_joint = cmds.listRelatives(child_joint, parent=True, fullPath=True)[0]
    if(("twist" in parent_joint.rsplit('|', 1)[-1]) or ("Twist" in parent_joint.rsplit('|', 1)[-1])):
        parent_joint = cmds.listRelatives(parent_joint, parent=True, fullPath=True)[0]
    parent_delta_global_rot = get_parent_delta_global_rot(parent_joint)
    
    parent_pos = get_joint_position(parent_joint)
    child_pos = get_joint_position(child_joint)
    boneVec = om.MVector(child_pos[0] - parent_pos[0], child_pos[1] - parent_pos[1],child_pos[2] - parent_pos[2])
    targetVec = om.MVector(float(target_vector[0]), float(target_vector[1]), float(target_vector[2]))
    needRot = boneVec.rotateTo(targetVec)
    needRotMat = needRot.asMatrix()

    jo_mat = getJointOrient(parent_joint)
    ra_mat = getRotateAxisRot(parent_joint)
    current_global = getGlobalRot(parent_joint)

    tmp1 = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp2 = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp3 = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    new_rot_mat = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp1.setToProduct(jo_mat,parent_delta_global_rot)
    tmp2.setToProduct(current_global,needRotMat)
    tmp3.setToProduct(ra_mat.inverse(),tmp2)
    new_rot_mat.setToProduct(tmp3,tmp1.inverse())

    new_rot_transMat =  om.MTransformationMatrix(new_rot_mat)
    new_rot = new_rot_transMat.rotation(asQuaternion=True)
    new_rot_euler = new_rot.asEulerRotation()

    order = getRotOrder(parent_joint)
    ordered_euler = new_rot_euler.reorderIt(order)
    # 应用旋转到父关节的局部旋转
    print(f"Adjust {parent_joint}")
    cmds.setAttr(parent_joint + '.rotateX', math.degrees(ordered_euler[0]))
    cmds.setAttr(parent_joint + '.rotateY', math.degrees(ordered_euler[1]))
    cmds.setAttr(parent_joint + '.rotateZ', math.degrees(ordered_euler[2]))

def find_joint_by_base_name(base_name, root_joint):
    """在指定根关节下查找具有相同基名称的关节"""
    all_joints = cmds.listRelatives(root_joint, allDescendents=True, type="joint") or []
    all_joints.append(root_joint)
    
    for joint in all_joints:
        if get_base_name(joint) == base_name:
            return joint
    return None

def get_joint_direction(joint_name):
    """获取关节相对于父关节的方向向量"""
    parent = cmds.listRelatives(joint_name, parent=True)
    
    if not parent:
        return [0, 0, 0]
    
    pos1 = cmds.xform(joint_name, q=True, ws=True, t=True)    
    child_trans = om.MVector(pos1)
    pos2 = cmds.xform(parent[0], q=True, ws=True, t=True)
    parent_trans = om.MVector(pos2)
    child_pos = [child_trans[0],child_trans[1],child_trans[2]]
    parent_pos = [parent_trans[0],parent_trans[1],parent_trans[2]]
    
    return [
        child_pos[0] - parent_pos[0],
        child_pos[1] - parent_pos[1],
        child_pos[2] - parent_pos[2]
    ]

def resetTwist(jointname, original_rot, rotAxis):
    rotAixsName = ".rotateX"
    rotIndex = 0
    if(rotAxis==[1,0,0]):
        rotAixsName = ".rotateX"
        rotIndex = 0
    if(rotAxis==[0,1,0]):
        rotAixsName = ".rotateY"
        rotIndex = 1
    if(rotAxis==[0,0,1]):
        rotAixsName = ".rotateZ"
        rotIndex = 2    
    cmds.setAttr(jointname + rotAixsName, original_rot[rotIndex])

def align_skeleton(joint_map, mh_root, daz_root):
    """对齐骨架的主函数"""
    processed_parents = set()
    error_joints = []
    joint_aim_map = {}
    joint_local_rot = {}

    for mh_joint_name, daz_joint_name in joint_map.items():
        daz_joint = find_joint_by_base_name(daz_joint_name, daz_root)
        parent = cmds.listRelatives(daz_joint, parent=True)
        joint_aim_map[parent[0]] = getAimUpVector(parent[0],get_joint_direction(daz_joint))[0]
        joint_local_rot[parent[0]] = getlocalRotation(parent[0])

    for mh_joint_name, daz_joint_name in joint_map.items():
        # 查找MetaHuman关节
        mh_joint = find_joint_by_base_name(mh_joint_name, mh_root)
        if not mh_joint:
            cmds.warning(f"⚠️ MetaHuman joint '{mh_joint_name}' not found under root '{get_base_name(mh_root)}'")
            error_joints.append(mh_joint_name)
            continue
            
        # 查找DAZ关节
        daz_joint = find_joint_by_base_name(daz_joint_name, daz_root)
        if not daz_joint:
            cmds.warning(f"⚠️ DAZ joint '{daz_joint_name}' not found under root '{get_base_name(daz_root)}'")
            error_joints.append(daz_joint_name)
            continue
        
        # 获取DAZ父关节
        daz_parents = cmds.listRelatives(daz_joint, parent=True)
        if not daz_parents:
            cmds.warning(f"⚠️ DAZ joint '{get_base_name(daz_joint)}' has no parent")
            continue
        daz_parent = daz_parents[0]
        
        # 避免多次处理同一父关节
        if daz_parent in processed_parents:
            continue
            
        # 获取方向向量
        target_direction = get_joint_direction(mh_joint)        
        
        # 应用旋转
        try:
            # rotate_joint_to_direction(daz_parent,joint_aim_map[daz_parent],target_direction)
            # resetTwist(daz_parent,joint_local_rot[daz_parent],joint_aim_map[daz_parent])
            align_parent_to_vector(daz_joint,target_direction)
            processed_parents.add(daz_parent)
            cmds.warning(f"✅ Successfully aligned: {get_base_name(daz_parent)}")
        except Exception as e:
            cmds.warning(f"⚠️ Failed to align {get_base_name(daz_parent)}: {str(e)}")
            error_joints.append(mh_joint_name)
    
    # 显示总结信息
    if error_joints:
        cmds.warning(f"⚠️ Alignment completed with {len(error_joints)} errors.")
    else:
        cmds.warning("🎉 Alignment completed successfully!")

def save_joint_map(file_path, joint_map):
    """保存关节映射到JSON文件"""
    try:
        with open(file_path, 'w') as f:
            json.dump(joint_map, f, indent=4)
        return True
    except Exception as e:
        cmds.warning(f"⚠️ Failed to save joint map: {str(e)}")
        return False

def load_joint_map(file_path):
    """从JSON文件加载关节映射"""
    try:
        if not os.path.exists(file_path):
            return {}
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 兼容旧格式处理
        if isinstance(data, list):
            # 旧格式: [{"mh_joint": "name", "daz_joint": "name"}]
            new_data = {}
            for item in data:
                if isinstance(item, dict) and "mh_joint" in item:
                    mh_name = item["mh_joint"]
                    new_data[mh_name] = item.get("daz_joint", "")
            return new_data
        
        # 新格式: {"mh_joint_name": "daz_joint_name"}
        return data
    except Exception as e:
        cmds.warning(f"⚠️ Failed to load joint map: {str(e)}")
        return {}

def auto_detect_namespace(joint_list):
    """自动检测并返回最常见的命名空间"""
    namespaces = {}
    for joint in joint_list:
        base_name = get_base_name(joint)
        # 检测命名空间格式 (mynamespace:joint)
        if ":" in joint:
            ns = joint.split(":")[0]
            namespaces[ns] = namespaces.get(ns, 0) + 1
        # 检测前缀格式 (prefix_joint)
        else:
            prefix_match = re.search(r"^([^_]+)_", base_name)
            if prefix_match:
                prefix = prefix_match.group(1)
                namespaces[prefix] = namespaces.get(prefix, 0) + 1
    
    # 返回最常见的命名空间/前缀
    if namespaces:
        return max(namespaces, key=namespaces.get)
    return ""

_syncing_selection = False
def sync_selections(*_):
    """同步两个列表的选择状态"""
    global _syncing_selection
    
    # 如果正在同步中，则跳过
    if _syncing_selection:
        return
    # 设置同步标志
    _syncing_selection = True
    
    # 获取当前活动列表
    active_list = cmds.textScrollList("mhJointsList", q=True, selectItem=True)
    if active_list:
        # MetaHuman列表被选中
        selected_indices = cmds.textScrollList("mhJointsList", q=True, selectIndexedItem=True) or []
        if selected_indices:
            # 同步到DAZ列表
            cmds.textScrollList("dazJointsList", e=True, deselectAll=True)
            for index in selected_indices:
                cmds.textScrollList("dazJointsList", e=True, selectIndexedItem=index)
    else:
        # DAZ列表被选中
        selected_indices = cmds.textScrollList("dazJointsList", q=True, selectIndexedItem=True) or []
        if selected_indices:
            # 同步到MetaHuman列表
            cmds.textScrollList("mhJointsList", e=True, deselectAll=True)
            for index in selected_indices:
                cmds.textScrollList("mhJointsList", e=True, selectIndexedItem=index)
    
    # 清除同步标志
    _syncing_selection = False

def create_alignment_ui():
    """创建对齐工具的UI界面"""
    win_name = "skeletonAlignmentUI"
    if cmds.window(win_name, exists=True):
        cmds.deleteUI(win_name)
    
    # 创建主窗口 - 固定大小确保所有内容可见
    cmds.window(win_name, title="Skeleton Alignment Tool", width=520, height=700, sizeable=False)
    
    # 主布局 - 垂直列布局
    main_layout = cmds.columnLayout(
        adjustableColumn=True,
        columnAttach=('both', 5),
        rowSpacing=10,
        height=690
    )
    
    # ================ 骨架设置部分 ================
    skeleton_frame = cmds.frameLayout(
        label="Skeleton Settings",
        collapsable=True,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    
    # MetaHuman根关节设置
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="MetaHuman Root:", align="right")
    mh_root_field = cmds.textField("mhRootField", width=120)
    cmds.button(
        label="Get Selected", 
        annotation="Select MetaHuman root joint and click",
        command=lambda *_: cmds.textField(
            mh_root_field, 
            e=True, 
            text=cmds.ls(selection=True)[0] if cmds.ls(selection=True) else ""
        )
    )
    cmds.setParent("..")
    
    # DAZ根关节设置
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="DAZ Root Joint:", align="right")
    daz_root_field = cmds.textField("dazRootField", width=120)
    cmds.button(
        label="Get Selected", 
        annotation="Select DAZ root joint and click",
        command=lambda *_: cmds.textField(
            daz_root_field, 
            e=True, 
            text=cmds.ls(selection=True)[0] if cmds.ls(selection=True) else ""
        )
    )
    cmds.setParent("..")
    
    # DAZ命名空间
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="DAZ Namespace:", align="right")
    daz_ns_field = cmds.textField("dazNsField", width=120)
    cmds.button(
        label="Auto Detect", 
        command=lambda *_: auto_detect_ns_cmd()
    )
    cmds.setParent("..")
    cmds.setParent("..")
    
    # ================ 文件设置部分 ================
    file_frame = cmds.frameLayout(
        label="File Settings",
        collapsable=True,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    
    # JSON文件设置
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="JSON File Path:", align="right")
    json_path_field = cmds.textField("jsonPathField", width=120)
    cmds.button(
        label="Browse...", 
        command=lambda *_: cmds.textField(
            json_path_field, 
            e=True, 
            text=(cmds.fileDialog2(
                fileMode=1,
                fileFilter="JSON Files (*.json)"
            ) or [""])[0]
        )
    )
    cmds.setParent("..")
    
    # 文件控制按钮
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.button(label="Load Map", command=lambda *_: load_map_cmd(), height=30)
    cmds.button(label="Save Map", command=lambda *_: save_map_cmd(), height=30)
    cmds.button(label="Reset Map", command=reset_table, height=30)
    cmds.setParent("..")
    cmds.setParent("..")
    
    # ================ 关节映射部分 ================
    map_frame = cmds.frameLayout(
        label="Joint Mapping",
        collapsable=False,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    
    # 标题行
    cmds.gridLayout(numberOfColumns=2, cellWidth=250)
    cmds.text(label="MetaHuman Joints", align="center", height=20)
    cmds.text(label="DAZ Joints", align="center", height=20)
    cmds.setParent("..")
    
    # 列表区域 - 保证有足够的高度
    cmds.rowLayout(numberOfColumns=2, height=200)
    mh_list = cmds.textScrollList("mhJointsList", allowMultiSelection=True, height=200, width=250, selectCommand=sync_selections)
    daz_list = cmds.textScrollList("dazJointsList", allowMultiSelection=True, height=200, width=250, selectCommand=sync_selections)
    cmds.setParent("..")
    
    # 控制按钮
    cmds.gridLayout(numberOfColumns=4, cellWidth=130, cellHeight=30)
    cmds.button(label="Add Selected", command=lambda *_:add_selected_cmd())
    cmds.button(label="Remove Selected", command=lambda *_:remove_selected_cmd())
    cmds.button(label="Auto-Detect All", command=lambda *_:auto_detect_cmd())
    cmds.button(label="Add Fingers", command=lambda *_:add_finger_mapping())
    cmds.setParent("..")
    cmds.setParent("..")
    
    # ================ 执行按钮 ================
    cmds.separator(height=10)
    cmds.button(
        label="ALIGN SKELETONS", 
        height=50, 
        backgroundColor=[0.3, 0.6, 0.8],
        command=lambda *_: execute_alignment_cmd()
    )
    
    # 初始化表格和默认值
    reset_table()
    
    cmds.showWindow(win_name)

def auto_detect_ns_cmd():
    """自动检测并设置DAZ命名空间"""
    daz_root = cmds.textField("dazRootField", q=True, text=True)
    
    if not daz_root or not cmds.objExists(daz_root):
        cmds.warning("⚠️ Set the DAZ root joint first")
        return
    
    all_joints = cmds.listRelatives(daz_root, allDescendents=True, type="joint") or []
    all_joints.append(daz_root)
    
    ns = auto_detect_namespace(all_joints)
    
    if ns:
        cmds.textField("dazNsField", e=True, text=ns)
        cmds.warning(f"🔍 Detected DAZ namespace/prefix: {ns}")
    else:
        cmds.warning("⚠️ No common DAZ namespace/prefix found")

def load_map_cmd():
    """加载关节映射文件"""
    json_path = cmds.textField("jsonPathField", q=True, text=True)
    if not json_path or not os.path.exists(json_path):
        cmds.warning("⚠️ Select a valid JSON file path")
        return
    
    joint_map = load_joint_map(json_path)
    cmds.textScrollList("mhJointsList", e=True, removeAll=True)
    cmds.textScrollList("dazJointsList", e=True, removeAll=True)
    
    for mh_joint, daz_joint in joint_map.items():
        cmds.textScrollList("mhJointsList", e=True, append=mh_joint)
        cmds.textScrollList("dazJointsList", e=True, append=daz_joint)
    
    cmds.warning(f"📖 Loaded {len(joint_map)} mappings from {json_path}")

def save_map_cmd():
    """保存关节映射到文件"""
    json_path = cmds.textField("jsonPathField", q=True, text=True)
    if not json_path:
        cmds.warning("⚠️ Set a JSON file path first")
        return
    
    # 获取映射数据
    joint_map = {}
    mh_joints = cmds.textScrollList("mhJointsList", q=True, allItems=True) or []
    daz_joints = cmds.textScrollList("dazJointsList", q=True, allItems=True) or []
    
    if len(mh_joints) != len(daz_joints):
        cmds.warning("⚠️ MetaHuman and DAZ joint lists must be the same length")
        return
    
    for i in range(len(mh_joints)):
        joint_map[mh_joints[i]] = daz_joints[i]
    
    # 保存到文件
    if save_joint_map(json_path, joint_map):
        cmds.warning(f"💾 Saved {len(joint_map)} mappings to {json_path}")

def reset_table():
    """重置映射表到默认状态"""
    cmds.textScrollList("mhJointsList", e=True, removeAll=True)
    cmds.textScrollList("dazJointsList", e=True, removeAll=True)
    
    # 添加默认值
    defaults = [
        ("upperarm_l", "l_upperarm"),
        ("lowerarm_l", "l_forearm"),
        ("hand_l", "l_hand"),
        ("upperarm_r", "r_upperarm"),
        ("lowerarm_r", "r_forearm"),
        ("hand_r", "r_hand"),
    ]
    
    for mh, daz in defaults:
        cmds.textScrollList("mhJointsList", e=True, append=mh)
        cmds.textScrollList("dazJointsList", e=True, append=daz)
    
    cmds.warning("🔄 Reset to default mappings")

def add_selected_cmd():
    """添加选中的关节对到映射表"""
    selected = cmds.ls(selection=True, type="joint")
    if not selected or len(selected) < 2:
        cmds.warning("⚠️ Select 2 joints (MetaHuman joint first, then DAZ joint)")
        return
    
    # 获取基名称
    mh_joint = selected[0]
    daz_joint = selected[1]
    
    mh_base = get_base_name(mh_joint)
    daz_base = get_base_name(daz_joint)
    
    # 添加到列表
    cmds.textScrollList("mhJointsList", e=True, append=mh_base)
    cmds.textScrollList("dazJointsList", e=True, append=daz_base)

def remove_selected_cmd(*_):
    """从映射表中移除选中的关节对"""
    # 获取选中的项目
    mh_selected = cmds.textScrollList("mhJointsList", q=True, selectItem=True) or []
    daz_selected = cmds.textScrollList("dazJointsList", q=True, selectItem=True) or []
    
    # 如果没有选中任何项目，直接返回
    if not mh_selected and not daz_selected:
        return
    
    # 获取所有项目
    mh_items = cmds.textScrollList("mhJointsList", q=True, allItems=True) or []
    daz_items = cmds.textScrollList("dazJointsList", q=True, allItems=True) or []
    
    # 创建要删除的项目列表
    items_to_remove = set()
    
    # 添加MetaHuman选中的项目
    for item in mh_selected:
        if item in mh_items:
            items_to_remove.add(item)
    
    # 添加DAZ选中的项目
    for item in daz_selected:
        if item in daz_items:
            items_to_remove.add(item)
    
    # 如果没有任何项目可以删除，直接返回
    if not items_to_remove:
        return
    
    # 删除MetaHuman列表中的项目
    for item in items_to_remove:
        if item in mh_items:
            cmds.textScrollList("mhJointsList", e=True, removeItem=item)
    
    # 删除DAZ列表中的项目
    for item in items_to_remove:
        if item in daz_items:
            cmds.textScrollList("dazJointsList", e=True, removeItem=item)

def add_finger_mapping():
    """添加手指关节的默认映射"""
    finger_types = ["thumb", "index", "middle", "ring", "pinky"]
    
    for side in ["_l", "_r"]:
        for finger in finger_types:
            for i in range(1, 4):
                mh_joint = f"{finger}_0{i}{side}"
                daz_joint = f"{side[1:]}_{finger}{i}"
                cmds.textScrollList("mhJointsList", e=True, append=mh_joint)
                cmds.textScrollList("dazJointsList", e=True, append=daz_joint)
    
    cmds.warning("🖐 Added finger mappings")

def auto_detect_cmd():
    """自动检测关节映射"""
    # 获取根关节
    mh_root = cmds.textField("mhRootField", q=True, text=True)
    daz_root = cmds.textField("dazRootField", q=True, text=True)
    daz_ns = cmds.textField("dazNsField", q=True, text=True) or ""
    
    if not mh_root or not cmds.objExists(mh_root):
        cmds.warning("⚠️ Set the MetaHuman root joint first")
        return
        
    if not daz_root or not cmds.objExists(daz_root):
        cmds.warning("⚠️ Set the DAZ root joint first")
        return
    
    # 重置表格
    cmds.textScrollList("mhJointsList", e=True, removeAll=True)
    cmds.textScrollList("dazJointsList", e=True, removeAll=True)
    
    # 获取所有MetaHuman关节
    mh_joints = cmds.listRelatives(mh_root, allDescendents=True, type="joint", fullPath=True) or []
    mh_joints.append(mh_root)
    
    # 获取所有DAZ关节
    daz_joints = cmds.listRelatives(daz_root, allDescendents=True, type="joint", fullPath=True) or []
    daz_joints.append(daz_root)
    
    matched_count = 0
    
    # 创建映射 - 基于基名称匹配
    for mh_joint in mh_joints:
        mh_base = get_base_name(mh_joint)
        
        # 在DAZ关节中查找匹配项
        for daz_joint in daz_joints:
            daz_base = get_base_name(daz_joint)
            
            # 应用命名空间
            if daz_ns and daz_base.startswith(daz_ns):
                clean_name = daz_base[len(daz_ns):]
            else:
                clean_name = daz_base
            
            if mh_base == clean_name:
                cmds.textScrollList("mhJointsList", e=True, append=mh_base)
                cmds.textScrollList("dazJointsList", e=True, append=daz_base)
                matched_count += 1
                break
    
    cmds.warning(f"🔍 Auto-detected {matched_count} joint mappings")

def execute_alignment_cmd():
    """执行骨架对齐操作"""
    # 获取根关节
    mh_root = cmds.textField("mhRootField", q=True, text=True)
    daz_root = cmds.textField("dazRootField", q=True, text=True)
    
    if not mh_root or not cmds.objExists(mh_root):
        cmds.warning("⚠️ Set the MetaHuman root joint first")
        return
    
    if not daz_root or not cmds.objExists(daz_root):
        cmds.warning("⚠️ Set the DAZ root joint first")
        return
    
    # 构建关节映射
    joint_map = {}
    mh_joints = cmds.textScrollList("mhJointsList", q=True, allItems=True) or []
    daz_joints = cmds.textScrollList("dazJointsList", q=True, allItems=True) or []
    
    if not mh_joints or not daz_joints:
        cmds.warning("⚠️ No joint mappings found")
        return
    
    if len(mh_joints) != len(daz_joints):
        cmds.warning("⚠️ MetaHuman and DAZ joint lists must be the same length")
        return
    
    for i in range(len(mh_joints)):
        joint_map[mh_joints[i]] = daz_joints[i]
    
    # 自动保存当前映射（可选）
    json_path = cmds.textField("jsonPathField", q=True, text=True)
    if json_path:
        if save_joint_map(json_path, joint_map):
            cmds.warning(f"💾 Map saved to {json_path}")
    
    # 执行对齐
    cmds.warning("🔧 Starting skeleton alignment...")
    cmds.refresh(suspend=True)  # 提高性能
    align_skeleton(joint_map, mh_root, daz_root)
    cmds.refresh(suspend=False)
    cmds.refresh()
    cmds.warning("✅ Done!")

# 启动UI
if __name__ == "__main__":
    create_alignment_ui()