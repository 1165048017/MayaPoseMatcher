import maya.cmds as cmds
import maya.api.OpenMaya as om
import math

def printMatrix(matrix):
    matrix_4x4 = [
    [matrix[0], matrix[1], matrix[2], matrix[3]],  # 第1行
    [matrix[4], matrix[5], matrix[6], matrix[7]],  # 第2行
    [matrix[8], matrix[9], matrix[10], matrix[11]], # 第3行
    [matrix[12], matrix[13], matrix[14], matrix[15]] # 第4行
    ]

    print("4x4 矩阵格式：")
    for row in matrix_4x4:
        print(" ".join([f"{x:.6f}" for x in row]))

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

def draw_local_axes(joint, length=100.0):
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)
    
    m_matrix = getGlobalRot(joint)

    x_axis = om.MVector(m_matrix[0], m_matrix[1], m_matrix[2]).normal()
    y_axis = om.MVector(m_matrix[4], m_matrix[5], m_matrix[6]).normal()
    z_axis = om.MVector(m_matrix[8], m_matrix[9], m_matrix[10]).normal()
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length

    x_curve = cmds.curve(d=1, p=[(origin.x, origin.y, origin.z), (x_end.x, x_end.y, x_end.z)], k=[0,1])
    y_curve = cmds.curve(d=1, p=[(origin.x, origin.y, origin.z), (y_end.x, y_end.y, y_end.z)], k=[0,1])
    z_curve = cmds.curve(d=1, p=[(origin.x, origin.y, origin.z), (z_end.x, z_end.y, z_end.z)], k=[0,1])

    cmds.setAttr(x_curve + ".overrideEnabled", 1)
    cmds.setAttr(x_curve + ".overrideColor", 13)  # Red
    cmds.setAttr(y_curve + ".overrideEnabled", 1)
    cmds.setAttr(y_curve + ".overrideColor", 14)  # Green
    cmds.setAttr(z_curve + ".overrideEnabled", 1)
    cmds.setAttr(z_curve + ".overrideColor", 6)   # Blue

    return [x_curve, y_curve, z_curve]

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

def draw_local_axes2(joint, length=100.0):
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)

    parent_joint_name = cmds.listRelatives(joint, parent=True)[0]
    parent_delta_global_rot = get_parent_delta_global_rot(parent_joint_name)
        
    m_matrix= getInherentRotWithoutParent(joint)
    result_matrix=om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    result_matrix.setToProduct(m_matrix, parent_delta_global_rot)

    # 提取旋转轴方向向量
    x_axis = om.MVector(result_matrix[0], result_matrix[1], result_matrix[2]).normal()
    y_axis = om.MVector(result_matrix[4], result_matrix[5], result_matrix[6]).normal()
    z_axis = om.MVector(result_matrix[8], result_matrix[9], result_matrix[10]).normal()
    # 计算终点
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length

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

    return [x_curve, y_curve, z_curve]


def draw_joint_global_lra(joint, length=100.0):
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)
    
    # 获取 jointOrient 并转换为旋转矩阵
    rot_matrix = getJointOrient(joint)
    euler_matrix = getRotateAxisRot(joint)
    result_matrix=om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    result_matrix.setToProduct(euler_matrix,rot_matrix)

    # 提取旋转轴方向向量
    x_axis = om.MVector(result_matrix[0], result_matrix[1], result_matrix[2]).normal()
    y_axis = om.MVector(result_matrix[4], result_matrix[5], result_matrix[6]).normal()
    z_axis = om.MVector(result_matrix[8], result_matrix[9], result_matrix[10]).normal()
    # 计算终点
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length

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

    return [x_curve, y_curve, z_curve]


def draw_joint_lra(joint, length=100.0):
    # 获取关节世界位置
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)

    # 获取 jointOrient 并转换为旋转矩阵
    rot_matrix = getJointOrient(joint)

    # 提取旋转轴方向向量
    x_axis = om.MVector(rot_matrix[0], rot_matrix[1], rot_matrix[2]).normal()
    y_axis = om.MVector(rot_matrix[4], rot_matrix[5], rot_matrix[6]).normal()
    z_axis = om.MVector(rot_matrix[8], rot_matrix[9], rot_matrix[10]).normal()
    # 计算终点
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length

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

    return [x_curve, y_curve, z_curve]

    
selection = "lShin"#cmds.ls(selection=True, type='joint')
if selection:
    #draw_joint_lra(selection) #仅仅绘制关节方向joint Orient
    #draw_joint_global_lra(selection) # 绘制关节方向*旋转轴;注意此时如果没有局部旋转,则结果应与局部旋转相同
    # draw_local_axes(selection) #绘制局部旋转
    draw_local_axes2(selection) #手动模拟绘制局部旋转
else:
    cmds.warning("请先选择一个关节。")