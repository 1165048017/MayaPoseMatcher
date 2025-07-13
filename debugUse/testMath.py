import numpy as np

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
    
    return theta

# 示例：计算两个三维向量的夹角
a = [1.1763650178909302, -16.392879486084, -44.7665405273438]  # 向量 a
b = [8.299, -8.034, 3.327]  # 向量 b

angle_radians = calculate_angle_3d(a, b)
angle_degrees = np.degrees(angle_radians)  # 转换为角度

print(f"夹角（弧度）: {angle_radians:.4f}")
print(f"夹角（角度）: {angle_degrees:.4f}")