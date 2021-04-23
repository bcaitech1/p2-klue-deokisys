# Pstage_03_KLUE_Relation_extraction - 서준배

## 성능평가
- aistages에서 진행한 `문장 내 개체간 관계 추출`의 test셋을 통한 성능평가 결과입니다.

### accuract
- 77.4%

### 최종순위
- 136/94

## 코드
- train.py
    - 학습이 진행됩니다.
- inference.py
    - test셋으로 학습한 모델을 평가합니다.
- load_data.py
    - 데이터를 로드하는데 필요한 함수가 모여져있습니다.

## 수정사항
- 코드내의 random시드 고정

### 사용한 모델
- `xlm-roberta-large`

### 변형된 파라메터
- epoch : 10
- batch-size : 32
- lr : 1e-5

### 입력값 변형
- 기존의 `단어[sep]단어[sep]문장`에서 문장내에 단어마다 `[e1]`,`[e2]`토큰으로 해당 단어를 특정할 수 있도록 수정하였습니다.
    - 기존의 단순문장이 뒤에 붙었을때, 앞에서 보여준 단어의 토큰이 문장을 토크나이징했을때 토큰과 달라지는 경우를 보았습니다.
    - 예를들어, `자동차`라는 단어가 하나의 토큰으로 만들어지는데 문장내에서는 `자동차의`라고 되어있어 토큰화하면 다른 토큰들로 토크나이저가 되었습니다.
    - 이런 문장 내부의 단어마다 토큰으로 감싸는 과정을 통해 좀더 성능 향상을 기대하였고 0.7정도의 성능향상을 보았습니다.

### 실험해본 내역 엑셀 링크
- [실험](https://docs.google.com/spreadsheets/d/1NTLhaM8UOH8hWqVgUPT09MTCL7VkJZZayDU6T-Tos6M/edit?usp=sharing)

## 설치 및 사용법

### 기본 설치
- `pip install -r requirements.txt`


### training
* `python train.py`

### inference
* `python inference.py --model_dir=[model_path]`
* ex) `python inference.py --model_dir=./results/checkpoint-500`

### evaluation
* `python eval_acc.py`
