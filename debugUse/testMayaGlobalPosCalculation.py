import maya.cmds as cmds
import numpy as np
import math
import maya.api.OpenMaya as om
np.set_printoptions(suppress=True, threshold=np.inf)

def calculate_angle_3d(a, b):
    """
    计算两个三维向量之间的夹角（弧度）
    
    参数:
        a (list or np.array): 第一个三维向量，如 [x1, y1, z1]
        b (list or np.array): 第二个三维向量，如 [x2, y2, z2]
    
    返回:
        float: 两向量的夹角（弧度）
    """
    # 转换为 numpy 数组
    a = np.array(a)
    b = np.array(b)
    
    # 计算点积
    dot_product = np.dot(a, b)
    
    # 计算向量的模
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_a * norm_b)
    
    # 计算夹角（弧度）
    theta = np.arccos(cos_theta)
    
    return np.degrees(theta)

def get_joint_orient(joint_name):
    joint_orient = cmds.getAttr(joint_name + ".jointOrient")
    return joint_orient

def get_joint_rotateAxis(joint_name):
    joint_axis = cmds.getAttr(joint_name + ".rotateAxis")
    return joint_axis
    
def get_global_info(joint_name):
    pos = cmds.xform(joint_name, q=True, ws=True, t=True)
    origin = om.MVector(pos)
    global_position = [origin[0],origin[1],origin[2]]
    if(True):
        # 获取关节的全局位置        
        matrix = cmds.xform(joint_name, q=True, m=True, ws=True)
        m_matrix = om.MMatrix(matrix)
        global_rotation = np.array([[m_matrix[0],m_matrix[1],m_matrix[2]],[m_matrix[4],m_matrix[5],m_matrix[6]],[m_matrix[8],m_matrix[9],m_matrix[10]]])
        return [global_position,global_rotation]
    else:
        
        matrix = cmds.xform(joint_name, q=True, m=True, ws=True)
        m_matrix = om.MMatrix(matrix)

        joint_axis = cmds.getAttr(joint_name + ".rotateAxis")[0]
        order = cmds.getAttr(joint_name + ".rotateOrder")
        euler = om.MEulerRotation(
            math.radians(joint_axis[0]),
            math.radians(joint_axis[1]),
            math.radians(joint_axis[2]),
            order
        )
        euler_matrix = euler.asMatrix()
        rot_matrix = m_matrix#*(euler_matrix.inverse())

        # 提取旋转轴方向向量
        x_axis = om.MVector(rot_matrix[0], rot_matrix[1], rot_matrix[2]).normal()
        y_axis = om.MVector(rot_matrix[4], rot_matrix[5], rot_matrix[6]).normal()
        z_axis = om.MVector(rot_matrix[8], rot_matrix[9], rot_matrix[10]).normal()
        # 计算终点
        x_end = origin + x_axis * 5
        y_end = origin + y_axis * 5
        z_end = origin + z_axis * 5

        # 绘制曲线
        x_curve = cmds.curve(d=1, p=[(origin.x, origin.y, origin.z), (x_end.x, x_end.y, x_end.z)], k=[0,1])
        y_curve = cmds.curve(d=1, p=[(origin.x, origin.y, origin.z), (y_end.x, y_end.y, y_end.z)], k=[0,1])
        z_curve = cmds.curve(d=1, p=[(origin.x, origin.y, origin.z), (z_end.x, z_end.y, z_end.z)], k=[0,1])

        # 设置颜色（红X，绿Y，蓝Z）
        cmds.setAttr(x_curve + ".overrideEnabled", 1)
        cmds.setAttr(x_curve + ".overrideColor", 13)
        cmds.setAttr(y_curve + ".overrideEnabled", 1)
        cmds.setAttr(y_curve + ".overrideColor", 14)
        cmds.setAttr(z_curve + ".overrideEnabled", 1)
        cmds.setAttr(z_curve + ".overrideColor", 6)
        
        global_rotation = np.array([[rot_matrix[0],rot_matrix[1],rot_matrix[2]],[rot_matrix[4],rot_matrix[5],rot_matrix[6]],[rot_matrix[8],rot_matrix[9],rot_matrix[10]]])
        return [global_position,global_rotation]
    
def drawCurve(start,end):
    # 绘制曲线
    x_curve = cmds.curve(d=1, p=[(start[0], start[1], start[2]), (end[0], end[1], end[2])], k=[0,1])

    # 设置颜色（红X，绿Y，蓝Z）
    cmds.setAttr(x_curve + ".overrideEnabled", 1)
    cmds.setAttr(x_curve + ".overrideColor", 13)

def get_local_info(joint_name):
    if(False):
        # 获取关节的全局位置
        local_position = cmds.xform(joint_name, query=True, worldSpace=False, translation=True)
        local_rotation = cmds.xform(joint_name, query=True, worldSpace=False, rotation=True)
        return [local_position,local_rotation]
    else:
        matrix = cmds.xform(joint_name, q=True, m=True, ws=False)
        m_matrix = om.MMatrix(matrix)
        local_position = [m_matrix[12],m_matrix[13],m_matrix[14]]
        local_rotation = np.array([[m_matrix[0],m_matrix[1],m_matrix[2]],[m_matrix[4],m_matrix[5],m_matrix[6]],[m_matrix[8],m_matrix[9],m_matrix[10]]])
        return [local_position,local_rotation]

def euler_to_rotation_matrix(euler_angles):
    # 将欧拉角转换为弧度
    euler_angles_rad = np.radians(euler_angles)

    # 计算旋转矩阵
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(euler_angles_rad[0]), -np.sin(euler_angles_rad[0])],
                           [0, np.sin(euler_angles_rad[0]), np.cos(euler_angles_rad[0])]])

    rotation_y = np.array([[np.cos(euler_angles_rad[1]), 0, np.sin(euler_angles_rad[1])],
                           [0, 1, 0],
                           [-np.sin(euler_angles_rad[1]), 0, np.cos(euler_angles_rad[1])]])

    rotation_z = np.array([[np.cos(euler_angles_rad[2]), -np.sin(euler_angles_rad[2]), 0],
                           [np.sin(euler_angles_rad[2]), np.cos(euler_angles_rad[2]), 0],
                           [0, 0, 1]])

    # 组合旋转矩阵
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix

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
# 替换为您场景中的实际关节名称
joint_name = "abdomenLower"
parent_joint_name = cmds.listRelatives(joint_name, parent=True)[0]

# 获取attribute属性
att_parent_orient = get_joint_orient(parent_joint_name) 
att_parent_axis = get_joint_rotateAxis(parent_joint_name)

# 获取并打印关节的全局位置
# [global_position_parent,global_rotation_parent] = get_global_info(parent_joint_name) #父关节的全局位置和全局旋转
# [global_position_child,global_rotation_child] = get_global_info(joint_name) #父关节的全局位置和全局旋转
# att_rotation_parent = euler_to_rotation_matrix(att_parent_orient[0])
# rotationMatrix = euler_to_rotation_matrix(global_rotation_parent) # 转换为旋转矩阵
[global_position_parent,global_rotation_parent] = get_global_info(parent_joint_name) #父关节的全局位置和全局旋转
[global_position_child,global_rotation_child] = get_global_info(joint_name) #父关节的全局位置和全局旋转
[local_position_child,local_rotation_child] = get_local_info(joint_name) # 子关节的局部信息
calculate_child_pos = global_position_parent + (np.dot(np.array(local_position_child).transpose(),global_rotation_parent)).transpose()
drawCurve(global_position_parent,calculate_child_pos)
# 计算jointOrient的轴与关节朝向的夹角
xyz_vec = get_joint_lra(parent_joint_name)
global_bone_vec=[global_position_child[0]-global_position_parent[0],global_position_child[1]-global_position_parent[1],global_position_child[2]-global_position_parent[2]]
angleVec = calculate_angle_3d(np.array(xyz_vec[0]),np.array(global_bone_vec))
print("-------start----------")
print(f"Global Position of {parent_joint_name}: {global_position_parent}")
print(f"Local position of {joint_name}: {local_position_child}")
print(f"Global Position of {joint_name}: {global_position_child}")
print(f"Calculate Global Position of {joint_name}: {calculate_child_pos}")
print(f"global_rotation_parent of {parent_joint_name}: {global_rotation_parent}")
print(f"{parent_joint_name}:{angleVec}")
print("-------finish----------")
