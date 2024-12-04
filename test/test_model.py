from src.utils.utility import Utility

ciFar100_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
    "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange",
    "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
    "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]

print("cl", len(ciFar100_labels))

train_list = Utility.find_filenames_with_keyword("data/raw/cifar/cifar-100", "train")
test_list = Utility.find_filenames_with_keyword("data/raw/cifar/cifar-100", "test")
dataset = [(file, 'train') for file in train_list] + [(file, 'test') for file in test_list]

for fi, set_type in dataset:
    dct = Utility.parse_pickle(fi)
    print(max(dct[b'coarse_labels']))
    print(max(dct[b'fine_labels']))
