# -ETRI-Emotion-Recognition

### 사용한 라이브러리 DownloadList.ipynb 를 실행시켜 일괄 다운로드 가능  
  
  

## 데이터 전처리 과정 및 결과 -> prepare_data.py 참조
  #### 과정
  1. 오탈자 확인 Sess12,Sess17,disqust 등
  2. 평가자별 가중치 세우기: 이는 과반수의 사람이 동일한 라벨로 평가한 데이터를 얼마나 많이 똑같이 평가했냐 Accuracy를 통해 계산 
  3. 평가자별 가중치를 곱하여 Total Evaluation을 재계산한다 
  #### 결과
  가중치가 소수이기 때문에 웬만해서는 동일 점수가 나오기 힘들어 Total Evaluation의 중복라벨들이 전부 제거되었다. 이를 new_data.csv로 저장
  
  
## 코드 실행방식에 대한 설명
  1. 우선 DownloadList.ipynb 를 실행시켜 라이브러리를 다운받음
  2. prepare_data.py를 실행시켜 사용하는 데이터셋인 new_data.csv 생성
  ### RoBERTa 모델 사용시 
  1. RoBERTa 파일 경로에 들어가 
  train.py 실행 -> Base model / train_multilabel.py 실행 -> 불균형을 해결을 위한 Loss 함수 이용
  -> 밑에 인자 값에 맞추어 실행
      
  ```python
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
  ``` 
  
  작동 예시
  ```
  python train.py --fold 0 
  ```
  권장 사양: 해당 모델은 RTX3090을 통해 훈련하였고, default값이 사용한 파라미터입니다. 따라서 더 작은 사양에서는 batchsize나 max_len을 줄여서 사용하는게 좋습니다
  ``` 예시
  python train.py --fold 0 --batch_size 16 max_len 128
  ```
  
  2. RoBERTa_2-fold.ipynb 실행: 2겹의 모델로 첫 모델은 가장 라벨의 수가 많은 neutral을 이진분류하고, 두번째 모델은 나머지 라벨을 분류 -> 나누어서 학습하여 학습 중 불균형에 의해 적은 수를 가지는 라벨들이 학습되지 못하는 것을 방지한 모델 
  ```python
  #Check Parameters
  fold = 0
  model_name = 'klue/roberta-base'
  BATCH_SIZE =64
  MAX_LEN =256
  EPOCHS = 30
  set_lr = 1e-4
  ```
  에 파라미터를 수정하여 전체 실행하면 알아서 학습을 하고 Score 기록 (F1-micro, F1-macro, Accuracy, Balance Accuracy)
  
  ### MultiModel ( RoBERTa + Wav2Vec) 사용시 
  baseline(text+audio).ipynb -> 베이스라인 모델
  multimodel_2fold.ipynb -> 2겹의 모델 방식을 멀티모델에 적용한 것 
  
  ```python
  #Check Parameters
  fold = 0
  model_name = 'klue/roberta-base'
  BATCH_SIZE =64
  MAX_LEN =256
  EPOCHS = 30
  set_lr = 1e-4
  ```
  ```python
  #Check Parameters
  fold = 0
  model_name = 'klue/roberta-base'
  BATCH_SIZE =64
  MAX_LEN =196
  MAX_WV_LEN = 4 * 16000
  EPOCHS = 30
  set_lr = 1e-4
  ```
  
  둘 다 RoBERTa_2-fold.ipynb 와 실행 방식이 동일: 파라미터 입력후 통째로 실행
  권장 사양: 해당 모델은 RTX3090 2개를 Data Parallel을 통해 훈련하였고, 위에 적힌 값이 사용한 파라미터입니다. 따라서 더 작은 사양에서는 batchsize나 max_len을 줄여서 사용하는게 좋습니다
 
      
      
