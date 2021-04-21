import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import random


# 시드 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 평가를 위한 metrics function.


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def train():
    # load model and tokenizer
    # "bert-base-multilingual-cased",kykim/bert-kor-base,roberta-large-mnli
    MODEL_NAME = "kykim/bert-kor-base"
    tokenizer = AutoTokenizer.from_pretrained(  # roberta기준
        MODEL_NAME)  # XLMRobertaTokenizer AutoTokenizer.from_pretrained(MODEL_NAME)
    # autoTokenizer로 자동으로 가져온다.

    # 데이터셋 나누기
    datas = pd.read_csv("/opt/ml/input/data/train/train.tsv",
                        delimiter='\t', header=None)
    train, val = train_test_split(datas, test_size=0.2, random_state=42)
    # train = pd.DataFrame(data=train)
    train.to_csv('/opt/ml/input/data/train/train_train.tsv',
                 sep='\t', header=None, index=False)
    val.to_csv('/opt/ml/input/data/train/train_val.tsv',
               sep='\t', header=None, index=False)

    # load dataset
    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    train_label = train_dataset['label'].values

    dev_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    dev_label = dev_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    bert_config = BertConfig.from_pretrained(
        MODEL_NAME)  # XLMRobertaConfig,BertConfig.from_pretrained(MODEL_NAME)
    # print(bert_config)

    # bert_config.vocab_size = 42004  # 4개 추가 안됨
    bert_config.num_labels = 42
    model = AutoModelForSequenceClassification.from_pretrained(  # XLMRobertaForSequenceClassification AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=bert_config)
    model.resize_token_embeddings(len(tokenizer))  # 새로운 토큰 추가로 임베딩 크기 조정
    # model = BertForSequenceClassification(bert_config)  # from_pretrained가져오기
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=3,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=4,              # total number of training epochs
        learning_rate=5e-5,               # learning_rate
        per_device_train_batch_size=8,  # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.

        per_device_eval_batch_size=8,   # batch size for evaluation
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,            # evaluation step.
        # dataloader_num_workers=4,
        # label_smoothing_factor=0.5

        # 제일 높은것만 저장한다고 한다.
        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy"

    )
    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()


def main():
    # seed고정
    seed_everything(42)
    # wandb설정
    wandb.login()
    wandb.init(project="bcai2-klue", name="kim-e4-b8_lr5e-5_notval",)

    train()


if __name__ == '__main__':
    main()
