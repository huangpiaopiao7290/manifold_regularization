# import asyncio
#
# from src.utils.utility import Utility
#
# async def test_move_file():
#     src = r'C:\piao_programs\py_programs\DeepLearningProject\Manifold_SmiLearn\data\processed\cifar-10\test\airplane\aeroplane_s_000002.png'
#     dst = r'C:\piao_programs\py_programs\DeepLearningProject\Manifold_SmiLearn\data\processed'
#     await Utility._move_file(src, dst)
#
# if __name__ == '__main__':
#     asyncio.run(test_move_file())
import os

# 假设当前工作目录是 manifold/src
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))

print(project_root)