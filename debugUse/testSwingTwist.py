import numpy as np
import maya.cmds as cmds

def rotation_matrix_to_twist_swing(R, axis):
    """
    将旋转矩阵分解为 twist 和 swing 分量。

    参数:
    R : numpy.ndarray
        3x3 旋转矩阵。
    axis : numpy.ndarray
        旋转轴，3x1 向量。

    返回:
    twist : numpy.ndarray
        twist 分量，3x1 向量。
    swing : numpy.ndarray
        swing 分量，3x1 向量。
    """
    # 确保轴是单位向量
    axis = axis / np.linalg.norm(axis)

    # 计算 twist 旋转矩阵
    # twist 旋转轴是参考轴，旋转角度是原始旋转矩阵在该轴上的旋转分量
    twist_angle = np.arccos(np.clip(np.trace(R) / 2 - 0.5, -1, 1))
    if np.allclose(twist_angle, 0):
        twist = np.zeros(3)
    else:
        # 计算 twist 旋转矩阵
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_twist = np.eye(3) + np.sin(twist_angle) * K + (1 - np.cos(twist_angle)) * np.dot(K, K)
        twist = twist_angle * axis

    # 计算 swing 旋转矩阵
    # swing 旋转矩阵是原始旋转矩阵 "减去" twist 旋转矩阵
    R_swing = np.dot(R, R_twist.T)

    # 从 swing 旋转矩阵提取 swing 向量
    swing_angle = np.arccos(np.clip(np.trace(R_swing) / 2 - 0.5, -1, 1))
    if np.allclose(swing_angle, 0):
        swing = np.zeros(3)
    else:
        # 计算 swing 旋转向量
        swing_axis = np.array([R_swing[2, 1] - R_swing[1, 2],
                               R_swing[0, 2] - R_swing[2, 0],
                               R_swing[1, 0] - R_swing[0, 1]])
        swing_axis = swing_axis / np.linalg.norm(swing_axis)
        swing = swing_angle * swing_axis

    return twist, swing

# 将 twist 向量转换为旋转矩阵
def vector_to_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R



# 示例
# if __name__ == "__main__":
#     # 定义一个旋转矩阵（绕 z 轴旋转 45 度）
#     angle = np.pi / 4
#     R = np.array([[np.cos(angle), -np.sin(angle), 0],
#                   [np.sin(angle), np.cos(angle), 0],
#                   [0, 0, 1]])
#     axis = np.array([0, 0, 1])  # z 轴

#     twist, swing = rotation_matrix_to_twist_swing(R, axis)
#     print("Twist:", twist)
#     print("Swing:", swing)



# 获取所选关节的名称
selected_joint = "mixamorig1:LeftArm"#cmds.ls(selection=True)[0]

# 获取关节的世界变换矩阵
world_matrix = cmds.getAttr(selected_joint + '.worldMatrix')

# 将 Maya 的矩阵格式转换为 NumPy 矩阵
# Maya 返回的是 16 个元素的列表，表示 4x4 矩阵
world_matrix_np = np.array(world_matrix).reshape(4, 4)

# 提取 3x3 的旋转矩阵
rotation_matrix = world_matrix_np[:3, :3]

axis = np.array([0, 0, 1])  # 根据需要修改旋转轴

# 调用函数
twist, swing = rotation_matrix_to_twist_swing(rotation_matrix, axis)


twist_angle = np.linalg.norm(twist)
if not np.allclose(twist_angle, 0):
    twist_axis = twist / twist_angle
    R_twist = vector_to_rotation_matrix(twist_axis, twist_angle)
    # 计算去除 twist 后的旋转矩阵
    R_no_twist = np.dot(rotation_matrix, R_twist.T)
else:
    R_no_twist = rotation_matrix

# 将旋转矩阵转换为 Maya 的欧拉角
# 这里需要将 R_no_twist 转换为 Maya 的欧拉角表示
# 由于 Maya 的欧拉角表示与 NumPy 的表示可能不同，需要根据实际情况进行调整
# 以下是一个简单的示例，假设 Maya 使用 XYZ 顺序的欧拉角
euler_angles = np.degrees(np.array([np.arctan2(R_no_twist[2, 1], R_no_twist[2, 2]),
                                    np.arctan2(-R_no_twist[2, 0], np.sqrt(R_no_twist[2, 1]**2 + R_no_twist[2, 2]**2)),
                                    np.arctan2(R_no_twist[1, 0], R_no_twist[0, 0])]))

# 设置关节的旋转
cmds.setAttr(selected_joint + '.rotateX', euler_angles[0])
cmds.setAttr(selected_joint + '.rotateY', euler_angles[1])
cmds.setAttr(selected_joint + '.rotateZ', euler_angles[2])
