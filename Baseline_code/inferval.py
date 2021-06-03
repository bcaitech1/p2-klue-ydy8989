from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, Trainer, \
    TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from kobert_tokenization import KoBertTokenizer


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []
    logits_list = []
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        logits_list.append(logits)
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    print(output_pred)
    # print(len(output_pred))
    print(logits_list)
    # asdfasdfasdf
    return logits_list#output_pred#np.array(output_pred).flatten()


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def main(args):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    TOK_NAME = "bert-base-multilingual-cased"
    # tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
    # TOK_NAME = 'monologg/kobert'
    # TOK_NAME = 'monologg/koelectra-base-v3-discriminator'

    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
    # tokenizer = KoBertTokenizer.from_pretrained(TOK_NAME)

    # load my model
    MODEL_NAME1 = args.model_dir1  # model dir.
    MODEL_NAME2 = args.model_dir2  # model dir.
    MODEL_NAME3 = args.model_dir3  # model dir.
    MODEL_NAME4 = args.model_dir4  # model dir.
    MODEL_NAME5 = args.model_dir5  # model dir.

    # fold_num = args.model_dir[-1]
    model1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME1)
    model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME2)
    model3 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME3)
    model4 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME4)
    model5 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME5)
    model1.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)
    model5.to(device)

    # load test datset
    test_dataset_dir = "../input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    logits_list1 = inference(model1, test_dataset, device)
    logits_list2 = inference(model2, test_dataset, device)
    logits_list3 = inference(model3, test_dataset, device)
    logits_list4 = inference(model4, test_dataset, device)
    logits_list5 = inference(model5, test_dataset, device)
    val_out_pred = []
    for val_iter in range(len(logits_list1)):
        val_logit = logits_list1[val_iter]+logits_list2[val_iter]+\
                    logits_list3[val_iter]+logits_list4[val_iter]+logits_list5[val_iter]
        result = np.argmax(val_logit, axis=-1)
        val_out_pred.append(result)
    kfold_answer = np.array(val_out_pred).flatten()

    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(kfold_answer, columns=['pred'])
    output.to_csv('./prediction/submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model dir
    parser.add_argument('--model_dir1', type=str, default="./results/base/1fold/checkpoint-565")
    parser.add_argument('--model_dir2', type=str, default="./results/base/2fold/checkpoint-565")
    parser.add_argument('--model_dir3', type=str, default="./results/base/3fold/checkpoint-565")
    parser.add_argument('--model_dir4', type=str, default="./results/base/4fold/checkpoint-565")
    parser.add_argument('--model_dir5', type=str, default="./results/base/5fold/checkpoint-565")

    args = parser.parse_args()
    print(args)
    main(args)
