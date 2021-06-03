import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *
from kobert_tokenization import KoBertTokenizer
import numpy as np
from torch.optim.lr_scheduler import StepLR

import random
# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# train_dataset, dev_dataset = train_test_split(dataset,test_size=0.2,shuffle=True)


def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-multilingual-cased"
    # MODEL_NAME = "monologg/distilkobert"
    # MODEL_NAME = 'monologg/kobert'
    MODEL_NAME = 'monologg/koelectra-base-v3-discriminator'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    # load dataset
    # train_dataset = load_data("../input/data/train/EDAtrain.tsv")
    train_dataset = load_data("../input/data/train/train2.tsv")
    # train_dataset = pd.read_csv("../input/data/train/train3.tsv",
    #                             usecols=['sentence','entity_01','entity_02','label'])
    train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
    leng = len(train_dataset)
    dev_dataset = train_dataset.iloc[int(0.8*leng):]
    train_dataset = train_dataset.iloc[:int(0.8*leng)]
    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    # bert_config = BertConfig.from_pretrained(MODEL_NAME)
    # bert_config.num_labels = 42
    # model = BertForSequenceClassification.from_pretrained(MODEL_NAME,config=bert_config)

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='../code/results',  # output directory
        overwrite_output_dir=True,
        save_total_limit=5,  # number of total save model.
        # save_steps=100,  # model saving step.
        num_train_epochs=5,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='../code/logs',  # directory for storing logs
        logging_steps=225,  # log saving step.
        evaluation_strategy='epoch', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        # eval_steps = 100,            # evaluation step.

        load_best_model_at_end = True
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        # optimizers=(optimizer,
                    # StepLR(optimizer, 1, gamma=0.794))
    )

    # train model
    trainer.train()


def main():

    train()

if __name__ == '__main__':
    main()
