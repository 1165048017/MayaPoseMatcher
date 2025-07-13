import maya.cmds as cmds
import maya.api.OpenMaya as om
import numpy as np
import math

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

'''
parent_delta_global_rot * (jo*anim*ra) = current_global

A->B NeedRot

NeedRot*current_global = parent_delta_global_rot * (jo*anim_new*ra)
(parent_delta_global_rot*jo).inv * (NeedRot*current_global) * ra.inv = anim_new
'''
def align_parent_to_vector(child_joint, target_vector):
    """将父关节指向子关节的方向对齐到目标向量"""
    # 获取父关节和子关节的位置
    parent_joint = cmds.listRelatives(child_joint, parent=True, fullPath=True)[0]
    if("twist" in parent_joint or "Twist" in parent_joint):
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
    print(new_rot_euler[0],new_rot_euler[1],new_rot_euler[2])
    print(ordered_euler[0],ordered_euler[1],ordered_euler[2])

    # 应用旋转到父关节的局部旋转
    cmds.setAttr(parent_joint + '.rotateX', math.degrees(ordered_euler[0]))
    cmds.setAttr(parent_joint + '.rotateY', math.degrees(ordered_euler[1]))
    cmds.setAttr(parent_joint + '.rotateZ', math.degrees(ordered_euler[2]))
    # 绘制代码
    draw_vector(parent_pos, target_vector, length=50.0, name="target_vector")

def draw_vector(origin, direction, length=50.0, color=(1, 0, 0), name="vector"):
    """绘制向量"""
    # 创建箭头曲线
    end_point = origin + normalize_vector(direction) * length
    arrow_curve = cmds.curve(d=1, p=[origin, end_point], n=name)
    # 设置曲线颜色
    cmds.setAttr(arrow_curve + '.overrideEnabled', 1)
    cmds.setAttr(arrow_curve + '.overrideColor', 13)  # 13是红色
    return arrow_curve

# 示例用法
if __name__ == "__main__":
    # 目标方向向量（可以根据需要修改）
    target_vector = np.array([0, 1, 0])

    # 获取当前选中的关节（子关节）
    selected_joints = cmds.ls(selection=True, type='joint')
    if selected_joints:
        child_joint = selected_joints[0]
        align_parent_to_vector(child_joint, target_vector)
    else:
        cmds.warning("请选中一个关节！")