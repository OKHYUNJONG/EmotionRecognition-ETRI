import os
import numpy as np
import pandas as pd


data_dir = '/workspace/ETRI/KEMDy20' # 당신의 경로를 넣어주세요. KEMDy20 폴더
data_dir2 = '/workspace/ETRI' #전체 폴더 데이터

# 잘못 표기된 두개의 데이터 수정
annotation_df = pd.read_csv(data_dir + '/annotation/Sess12_eval.csv')
for i in range(1,len(annotation_df)):
    tmp = annotation_df.loc[i,'Segment ID']
    annotation_df.loc[i,'Segment ID'] = 'Sess12' + tmp[6::]
annotation_df.to_csv(data_dir + '/annotation/Sess12_eval.csv',index=False)

annotation_df = pd.read_csv(data_dir + '/annotation/Sess17_eval.csv')
for i in range(1,len(annotation_df)):
    tmp = annotation_df.loc[i,'Segment ID']
    annotation_df.loc[i,'Segment ID'] = 'Sess17' + tmp[6::]
annotation_df.to_csv(data_dir + '/annotation/Sess17_eval.csv',index=False)

# 데이터하나로 합치기 + 텍스트 애트리뷰트 추가 
df = pd.DataFrame()
for num in range(1,41):
    if num < 10:
        num = "0" + str(num)
    else:
        num = str(num)

    annotation_df = pd.read_csv(data_dir + f'/annotation/Sess{num}_eval.csv')

    for i in range(1,len(annotation_df)):
        tmp_ID = annotation_df.loc[i,'Segment ID']
        tmp_ID = f"Sess{num}" + tmp_ID[6:]
        txt_path = data_dir + f'/wav/Session{num}/' + tmp_ID + '.txt'
        with open(txt_path, "r",encoding='cp949') as file:
            strings = file.read()
            annotation_df.loc[i,'text'] = strings
    df = pd.concat([df,annotation_df])

# 오타 수정
df['Total Evaluation'] = df['Total Evaluation'].str.replace('disqust', 'disgust')

# 평가자별 가중치 계산 및 라벨 재결정
df = df.dropna()
df = df.reset_index(drop=True)
df = df[df.columns[df.columns.str.contains('Eval')].tolist()]

targets = ['neutral',
    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise']
    
for target in targets:
    df[target] = 0

for i in range(len(df)):
    for j in range(1,11):
        for target in targets:
            if df.iloc[i,j] == target:
                df.loc[i,target] +=1

df['count'] = 0
for i in range(len(df)):
    max_count = 0
    for target in targets:
        if max_count < df.loc[i,target]:
            max_count = df.loc[i,target]

    df.loc[i,'count'] = max_count

# 과반수인 데이터만 추출하여 평가자별 가중치 계산
new_df = df[df['count'] >5]
new_df = new_df.reset_index(drop=True)

arr = [0 for i in range(10)]
for i in range(len(new_df)):
    for j in range(10):
        if new_df.iloc[i,j+1] in new_df.iloc[i,0]:
            arr[j] += 1

for i in range(len(arr)):
    arr[i] /= len(new_df)

for target in targets:
    df[target] = 0

for i in range(len(df)):
    for j in range(1,11):
        for target in targets:
            if df.iloc[i,j] == target:
                df.loc[i,target] += (arr[j-1] / sum(arr))

df['count'] = 0
for i in range(len(df)):
    max_count = 0
    for target in targets:
        if max_count < df.loc[i,target]:
            max_count = df.loc[i,target]

    df.loc[i,'count'] = max_count

cnt=[]
for i in range(len(df)):
    c = 0
    max_count = df.loc[i,'count']
    for target in targets:
        if df.loc[i,target] == max_count:
            tmp = target
            c +=1
    if c == 1:
        df.loc[i,'Total Evaluation'] = tmp
    else:
        cnt.append(i)

new_df = pd.read_csv(data_dir2 + '/data.csv')
new_df['Total Evaluation'] = df['Total Evaluation']

for target in targets:
    new_df[target] = df[target]

new_df['max_count'] = df['count']

#레이블 재평가된 데이터 (new_data.csv)
new_df.to_csv('new_data.csv',index=False)