import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
  def __init__(self, subjects, df, tokenizer, max_len):
    self.subjects = subjects
    self.df = df
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.subjects)
  def __getitem__(self, item):
    subject = str(self.subjects[item])
    target = self.df.loc[item,['neutral','not_neutral']].values.astype('float')
    encoding = self.tokenizer.encode_plus(
      subject,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding = 'max_length',
      truncation = True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'subject_text': subject,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.float32)
    }
def create_data_loader_binary(df, tokenizer, max_len, batch_size, shuffle_=False, valid=False):
  ds = SentimentDataset(
    subjects=df.text.to_numpy(),
    df=df,
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
    shuffle = shuffle_
  )