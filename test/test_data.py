import os

print(os.getcwd())

import os

# 定义路径
path = 'C:\\piao_programs\\py_programs\\DeepLearningProject\\Manifold_SmiLearn'

# 使用 os.path.split() 分割路径
head, tail = os.path.split(path)
print(head, tail)

# 再次分割以获取最后一个文件夹
last_folder = os.path.basename(os.path.dirname(path))
print(last_folder)  # 输出: folder2