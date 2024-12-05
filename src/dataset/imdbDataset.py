from torch.utils.data import Dataset

MAX_WORDS = 10000  # imdb’s vocab_size 即词汇表大小
MAX_LEN = 200      # max length
BATCH_SIZE = 256
EMB_SIZE = 128   # embedding size
HID_SIZE = 128   # lstm hidden size
DROPOUT = 0.2


class IMDBDataset(Dataset):
    def __init__(self) -> None:

