import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import argparse
import random
from sklearn.model_selection import StratifiedKFold
from Dataset_multilabel import * 
from utils import *
from Model import *
import torch.optim as optim
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import re

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples,epoch):

  batch_time = AverageMeter()     
  data_time = AverageMeter()      
  losses = AverageMeter()         
  accuracies = AverageMeter()
  f1_accuracies = AverageMeter()
  
  sent_count = AverageMeter()   
    

  start = end = time.time()

  model = model.train()
  correct_predictions = 0
  for step,d in enumerate(data_loader):
    data_time.update(time.time() - end)
    batch_size = d["input_ids"].size(0) 

    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    _, targets_max = torch.max(targets, dim=1)
    correct_predictions += torch.sum(preds == targets_max)
    losses.update(loss.item(), batch_size)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    batch_time.update(time.time() - end)
    end = time.time()

    sent_count.update(batch_size)
    if step % 50 == 0 or step == (len(data_loader)-1):
                acc,f1_acc = calc_SR_acc(outputs, targets_max)
                accuracies.update(acc, batch_size)
                f1_accuracies.update(f1_acc, batch_size)

                
                print('Epoch: [{0}][{1}/{2}] '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                      'Acc: {acc.val:.3f}({acc.avg:.3f}) '   
                      'f1_Acc: {f1_acc.val:.3f}({f1_acc.avg:.3f}) '           
                      'sent/s {sent_s:.0f} '
                      .format(
                      epoch, step+1, len(data_loader),
                      data_time=data_time, loss=losses,
                      acc=accuracies,
                      f1_acc=f1_accuracies,
                      remain=timeSince(start, float(step+1)/len(data_loader)),
                      sent_s=sent_count.avg/batch_time.avg
                      ))

  return correct_predictions.double() / n_examples, losses.avg

def validate(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
  model = model.eval()
  losses = []
  outputs_arr = []
  preds_arr = []
  targets_max_arr = []
  correct_predictions = 0
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    outputs_arr.append(outputs.cpu().detach().numpy()[0])
    preds_arr.append(preds.cpu().numpy())
    
    loss = loss_fn(outputs, targets)
    _, targets_max = torch.max(targets, dim=1)
    correct_predictions += torch.sum(preds == targets_max)
    targets_max_arr.append(targets_max.cpu().numpy())
    losses.append(loss.item())
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  return correct_predictions.double() / n_examples, np.mean(losses), outputs_arr, preds_arr, targets_max_arr


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#default 는 논문에서 사용한 것 -> 하이퍼파라미터 튜닝한 것은 아님
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str,default='klue/roberta-base', required=False)
    parser.add_argument("--lr", type=float,default=1e-4, required=False)
    parser.add_argument("--input", type=str, default="input", required=False)
    parser.add_argument("--max_len", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=30, required=False)
    parser.add_argument("--device", type=int, default=0, required=False)
    return parser.parse_args()

def remove_characters(sentence, lower=True):
    s = re.compile('\n')
    sentence = s.sub(' ', str(sentence))
    if lower:
        sentence = sentence.lower()
    return sentence


def main():
    args = parse_args()
    seed_everything(42)

    folds = [['Sess01','Sess02','Sess03','Sess04','Sess05','Sess06','Sess07','Sess08'],
    ['Sess09','Sess10','Sess11','Sess12','Sess13','Sess14','Sess15','Sess16'],
    ['Sess17','Sess18','Sess19','Sess20','Sess21','Sess22','Sess23','Sess24'],
    ['Sess25','Sess26','Sess27','Sess28','Sess29','Sess30','Sess31','Sess32'],
    ['Sess33','Sess34','Sess35','Sess36','Sess37','Sess38','Sess39','Sess40']]

    targets = ['neutral',
    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise']
    df = pd.read_csv('/workspace/ETRI/new_data.csv')
    df = df[['Numb','Segment ID','Total Evaluation','text','max_count'] + targets]
    df['text'] = df['text'].map(remove_characters)

    mapping_info = {"neutral":0,"happy":1,"surprise":2,"angry":3,"sad":4,"disgust":5,"fear":6}
    df['target'] = df['Total Evaluation'].map(mapping_info)
    test_list = '|'.join(folds[args.fold])
    test = df[df['Segment ID'].str.contains(test_list)]
    train =df[df['Segment ID'].str.contains(test_list) == False]


  
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_data_loader = create_data_loader(train, tokenizer, args.max_len, args.batch_size, shuffle_=True)
    test_data_loader = create_data_loader(test, tokenizer, args.max_len, args.valid_batch_size, valid=True)
    device = torch.device("cuda:"+ str(args.device))

    EPOCHS = args.epochs
    model = SentimentClassifier(n_classes=7, model_name = args.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps*0.1),
    num_training_steps=total_steps
    )
    
    nSamples = train.target.value_counts().tolist()
    num = 0
    for target in targets:
        nSamples[num] *=train[target].mean()
        num +=1
        
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)


    loss_fn = nn.MultiLabelSoftMarginLoss(weight=normedWeights).to(device)

    for epoch in range(EPOCHS):
        print('-' * 10)
        print(f'Epoch {epoch}/{EPOCHS-1}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train),
            epoch
        )
        validate_acc, validate_loss, outputs_arr, preds_arr, targets_max_arr= validate(
            model,
            test_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(test)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'validate loss {validate_loss} accuracy {validate_acc}')
        print(f'validate f1-score: ',f1_score(preds_arr, targets_max_arr, average='macro'))
        print("")
        print("")




if __name__ == "__main__":
    main()

