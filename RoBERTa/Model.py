from transformers import AutoModel
import torch.nn as nn

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes,model_name):
    super(SentimentClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained(model_name)
    self.drop = nn.Dropout(p=0.1)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def get_cls(target_size= n_classes):
      return nn.Sequential(
          nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
          nn.LayerNorm(self.bert.config.hidden_size),
          nn.Dropout(p = 0.1),
          nn.ReLU(),
          nn.Linear(self.bert.config.hidden_size, target_size),
      )  
    self.cls = get_cls(n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)

    out = self.cls(output)

    return out