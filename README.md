# KLUE:Relation extraction

## 🙋 작성자

- 윤도연 T1134

## 🏆최종 성적

- `Public LB` : Accuracy 79.9% | 46등



## 📚Task Description

`관계 추출(Relation Extraction)`은 문장의 `단어(Entity)`에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

- ***기간*** : 2021.04.12~2021.04.23(2주)

- ***Data description*** :

	- `input`: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.

	- `output`:relation 42개 classes 중 1개의 class를 예측한 값입니다.

		![image](https://user-images.githubusercontent.com/38639633/123445572-f9955180-d612-11eb-8b69-450686eb7202.png)

- ***Metric*** : Accuracy



## 📁프로젝트 구조

```
p2-klue-ydy8989>
├─Relation_Extraction
│   ├─config.yml
│   ├─evaluation.py
│   ├─infer.py
│   ├─kobert_tokenization.py
│   ├─load_data.py
│   ├─loss.py
│   ├─scheduler.py
│   ├─train.py
|   ├─requirements.txt
│   └─trainer_train.py
│
└─input
│   └─data
│       └─test
│           └─test.tsv
│       └─train
│           └─train.tsv
│       └─label_type.pkl

├─README.md
└─requirements.txt
```



### File overview

- `Relation_Extraction`

	- `evaluation.py` : evaluation print
	- `infer.py` : submission.csv를 만들기 위한 inference 파일
	- `kobert_tokenization.py` : kobert 사용에 필요한 클래스
	- `load_data.py` : 데이터 로더 및 전처리
	- `loss.py` : loss 함수 모음
	- `scheduler.py` : 스케줄러 모음
	- `trainer_train.py` : Huggingface의 trainer를 사용하는 버전의 train 파일
	- `train.py` : trainer를 사용하지 않는 버전의 train 파일

	

## :wink: Usage

### Training

```shell
# train.py
python train.py --config='roberta'

# trainer_train.py
python trainer_train.py --config='roberta'
```

- `config.yml`에서 하이퍼파라미터를 조정
- 해당 파라미터 버전을 `--config`로 파싱해 원하는 형태로 학습 가능합니다.



### Inference

```shell
# infer.py
python infer.py
```

