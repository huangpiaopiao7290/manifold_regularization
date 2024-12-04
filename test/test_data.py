import os

a = ("C:\\piao_programs\\py_programs\\DeepLearningProject"
     "\\Manifold_SmiLearn\\data\\processed\\cifar-100\\train\\label")
print(os.listdir(a))

samples = []
for label in os.listdir(a):
    label_dir = os.path.join(a, label)          # train/label/xxx
    # label = os.path.split(label_dir)[-1]
    for image_name in os.listdir(label_dir):
        image_path_labeled = os.path.join(label_dir, image_name)
        samples.append((image_path_labeled, -1))