import json
import re

# 创建数据结构
data = {
    "object1": {
        "name": "Cube",
        "position": [1, 2, 3],
        "scale": [4, 5, 6]
    },
    "object2": {
        "name": "Sphere",
        "position": [1, 2, 3],
        "scale": [4, 5, 6]
    }
}

data["object3"] = {
    "name": "Cylinder",
    "position": [7, 8, 9],
    "scale": [10, 11, 12]
}

# 自定义JSON编码器确保数组内不换行，并保持字段间换行
class CompactArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        # 首先使用默认编码器生成带缩进的JSON
        json_str = super().encode(obj)
        
        # 使用正则表达式移除数组内的换行和缩进
        json_str = re.sub(r'\[\s+', '[', json_str)
        json_str = re.sub(r'\s+\]', ']', json_str)
        json_str = re.sub(r',\s+', ',', json_str)
        
        # 修复字段间的换行问题
        # 确保每个字段结束后有换行
        json_str = re.sub(r'",\s*"', '",\n    "', json_str)
        # 确保对象结束后有换行
        json_str = re.sub(r'}\s*,', '},\n  ', json_str)
        json_str = re.sub(r'}\s*}', '}\n  }', json_str)

        json_str = re.sub(r']\s*,', '],\n    ', json_str)
        
        return json_str

# 生成JSON字符串（指定缩进和自定义编码器）
json_str = json.dumps(
    data,
    indent=2,
    cls=CompactArrayEncoder,
    separators=(',', ': ')  # 控制键值对分隔符
)

# 将JSON写入文件
with open('output.json', 'w') as f:
    f.write(json_str)

print("JSON文件已成功写入：output.json")


with open('output.json', 'r') as f:
    load_data = json.load(f)
print(load_data["object3"]["position"])