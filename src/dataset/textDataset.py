import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载并准备数据
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # 根据需要调整

imdb_df = pd.read_csv('path_to_imdb.csv')  # 假设 CSV 文件有 'text' 和 'label' 列
yahoo_df = pd.read_csv('path_to_yahoo.csv')

imdb_dataset = TextDataset(imdb_df, tokenizer, max_len)
yahoo_dataset = TextDataset(yahoo_df, tokenizer, max_len)

imdb_loader = DataLoader(imdb_dataset, batch_size=16, shuffle=True)
yahoo_loader = DataLoader(yahoo_dataset, batch_size=16, shuffle=True)