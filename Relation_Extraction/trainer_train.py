# python trainer_train.py --config config_name

import os
import yaml
import random
import argparse
import numpy as np
import torch

from transformers import *
from easydict import EasyDict
from prettyprinter import cpprint
from importlib import import_module
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer
from transformers import EarlyStoppingCallback

from load_data import *
from loss import *


# Set Config
class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()

    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()

    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)


# 시드 고정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # loss_fn = CustomLoss()
    # calculate accuracy using sklearn's function
    print(labels.shape, preds.shape)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    # loss = loss_fn(labels, preds)
    return {
        'accuracy': round(acc, 4),
        'f1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        # 'loss':round(loss,4),
    }


'''
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001)
'''


def train(cfg):
    SEED = cfg.values.seed
    MODEL_NAME = cfg.values.model_name
    USE_KFOLD = cfg.values.val_args.use_kfold
    TSVFILE = cfg.values.tsvfile
    #
    # early_stopping = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)
    # early_stopping_patience : 몇 번(epoch)을 참아줄 것인가?
    # early_stopping_threshold : metric이 어느 정도 개선 되어야 하는가?

    seed_everything(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # model_config_module = getattr(import_module('transformers'), cfg.values.model_arc + 'Config')
    # model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = ElectraConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42

    whole_df = load_data("/opt/ml/input/data/train/"+TSVFILE)
    whole_label = whole_df['label'].values
    # tokenizer_module = getattr(import_module('transformers'), cfg.values.model_arc + 'Tokenizer')
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=cfg.values.train_args.output_dir,  # output directory
        save_total_limit=cfg.values.train_args.save_total_limit,  # number of total save model.
        save_steps=cfg.values.train_args.save_steps,  # model saving step.
        num_train_epochs=cfg.values.train_args.num_epochs,  # total number of training epochs
        learning_rate=cfg.values.train_args.lr,  # learning_rate
        fp16=True,
        per_device_train_batch_size=cfg.values.train_args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=cfg.values.train_args.eval_batch_size,  # batch size for evaluation
        warmup_steps=cfg.values.train_args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=cfg.values.train_args.weight_decay,  # strength of weight decay
        logging_dir=cfg.values.train_args.logging_dir,  # directory for storing logs
        logging_steps=cfg.values.train_args.logging_steps,  # log saving step.
        evaluation_strategy=cfg.values.train_args.evaluation_strategy,  # evaluation strategy to adopt during training
        dataloader_num_workers=4,
        label_smoothing_factor=cfg.values.train_args.label_smoothing_factor,
        greater_is_better=True,
        metric_for_best_model = cfg.values.train_args.metric_for_best_model,
        # lr_scheduler_type='get_cosine_with_hard_restarts_schedule_with_warmup'
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=cfg.values.train_args.eval_steps,  # evaluation step.
        load_best_model_at_end=cfg.values.train_args.load_best_model_at_end
    )

    if USE_KFOLD:
        kfold = StratifiedKFold(n_splits=cfg.values.val_args.num_k)

        k = 1
        for train_idx, val_idx in kfold.split(whole_df, whole_label):
            print('\n')
            cpprint('=' * 15 + f'{k}-Fold Cross Validation' + '=' * 15)
            train_df = whole_df.iloc[train_idx]
            val_df = whole_df.iloc[val_idx]

            tokenized_train = tokenized_dataset(train_df, tokenizer)
            tokenized_val = tokenized_dataset(val_df, tokenizer)

            RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
            RE_val_dataset = RE_Dataset(tokenized_val, val_df['label'].values)

            # model_module = getattr(import_module('transformers'), cfg.values.model_arc + 'ForSequenceClassification')
            # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

            model.to(device)

            training_args.output_dir = cfg.values.train_args.output_dir + f'/{k}fold'
            training_args.logging_dir = cfg.values.train_args.output_dir + f'/{k}fold'

            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=RE_train_dataset,  # training dataset
                eval_dataset=RE_val_dataset,  # evaluation fkdataset
                compute_metrics=compute_metrics  # define metrics function
            )
            k += 1
            # train model
            trainer.train()
            if cfg.values.val_args.fold_break:
                break

    else:
        cpprint('=' * 20 + f'START TRAINING' + '=' * 20)

        train_df, val_df = train_test_split(whole_df, test_size=cfg.values.val_args.test_size, random_state=SEED)

        tokenized_train = tokenized_dataset(train_df, tokenizer)
        tokenized_val = tokenized_dataset(val_df, tokenizer)

        RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
        RE_val_dataset = RE_Dataset(tokenized_val, val_df['label'].values)

        # model_module = getattr(import_module('transformers'), cfg.values.model_arc + 'ForSequenceClassification')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        model.parameters
        model.to(device)

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function

        )

        # train model
        trainer.train()


def main(cfg):
    train(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='electra-small-v3')

    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)