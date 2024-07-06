#!/usr/bin/env python
# coding: utf-8
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import TrainerCallback, BertTokenizer, Trainer, BertPreTrainedModel, BertModel
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk, Dataset
import numpy as np

from function.collator_for_classification import DataCollatorForCellClassification

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='参数输入')
parser.add_argument('--input_dataset', type=str, help='输入数据集所在路径')
parser.add_argument('--input_model', type=str, help='输入模型所在路径')
parser.add_argument('--output_file', type=str, help='预测结果保存的路径')

args = parser.parse_args()
input_dataset = args.input_dataset
input_model = args.input_model
output_path = args.output_file

tokenizer = BertTokenizer(
    vocab_file=f'{input_model}/vocab_origin.txt', special_tokens={})

data = load_from_disk(input_dataset)


# filter dataset for given cell_type
def if_cell_type(example):
    valid_clusters = [
        'CD8_act_T_cells', 'CD8_ex_T_cells', 'CD8_mem_T_cells'
    ]
    return example["treatment"].startswith("pre") and example["cluster"] in valid_clusters


def pad_sequences_with_cls_token(sequences, max_length, pad_token_id, cls_token_id):
    padded_sequences = []
    attention_masks = []
    for seq in sequences:
        seq_len = len(seq)
        if (seq_len + 1) < max_length:
            pad_length = max_length - seq_len - 1
            padded_seq = [cls_token_id] + seq + [pad_token_id] * pad_length
            attention_mask = [1] * (seq_len + 1) + [0] * pad_length
        else:
            padded_seq = [cls_token_id] + seq[:max_length - 1]
            attention_mask = [1] * max_length

        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)

    return padded_sequences, attention_masks


# debug的时候为减少数据量，只取部分数据
data = data.filter(if_cell_type)
max_length = 2048
input_ids_padded_with_cls, attention_masks = pad_sequences_with_cls_token(data['input_ids'], max_length,
                                                                          tokenizer.pad_token_id,
                                                                          tokenizer.cls_token_id)
cell_ids = data['cell.id']
patients = data['patient']
treatments = data['treatment']
clusters = data['cluster']
responses = data['response']
responses = ["Yes" if s.startswith("Yes (CR)") else s for s in responses]
best_changes = data['best_change']

unique_patients = np.unique(patients)
train_patients, val_patients = train_test_split(unique_patients, test_size=0.2, random_state=16)
train_mask = np.isin(patients, train_patients)
val_mask = np.isin(patients, val_patients)
X_train = np.array(input_ids_padded_with_cls)[train_mask].tolist()
X_val = np.array(input_ids_padded_with_cls)[val_mask].tolist()
attention_train = np.array(attention_masks)[train_mask].tolist()
attention_val = np.array(attention_masks)[val_mask].tolist()
cell_ids_train = np.array(cell_ids)[train_mask].tolist()
cell_ids_val = np.array(cell_ids)[val_mask].tolist()
patients_train = np.array(patients)[train_mask].tolist()
patients_val = np.array(patients)[val_mask].tolist()
treatments_train = np.array(treatments)[train_mask].tolist()
treatments_val = np.array(treatments)[val_mask].tolist()
clusters_train = np.array(clusters)[train_mask].tolist()
clusters_val = np.array(clusters)[val_mask].tolist()
responses_train = np.array(responses)[train_mask].tolist()
responses_val = np.array(responses)[val_mask].tolist()
best_changes_train = np.array(best_changes)[train_mask].tolist()
best_changes_val = np.array(best_changes)[val_mask].tolist()

responses_array = np.array(responses)
patient_ids_array = np.array(patients)
treatment_ids_array = np.array(treatments)
unique_responses = np.unique(responses_array)
unique_patients = np.unique(patient_ids_array)
unique_treatments = np.unique(treatment_ids_array)


def classes_to_ids(example):
    example["label"] = dict(zip(unique_responses, [i for i in range(len(unique_responses))]))[example["label"]]
    example['patients'] = dict(zip(unique_patients, [i for i in range(len(unique_patients))]))[
        example['patients']]
    # example['treatment_ids'] = dict(zip(unique_treatments, [i for i in range(len(unique_treatments))]))[
    #     example['treatment_ids']]

    return example


train_dataset = Dataset.from_dict({
    'input_ids': X_train,
    'attention_mask': attention_train,
    'patients': patients_train,
    'treatments': treatments_train,
    'clusters': clusters_train,
    'cell_ids': cell_ids_train,
    'best_changes': best_changes_train,
    'label': responses_train
})
train_dataset_v2 = train_dataset.map(classes_to_ids).shuffle(seed=30)

val_dataset = Dataset.from_dict({
    'input_ids': X_val,
    'attention_mask': attention_val,
    'patients': patients_val,
    'treatments': treatments_val,
    'clusters': clusters_val,
    'cell_ids': cell_ids_val,
    'best_changes': best_changes_val,
    'label': responses_val
})
val_dataset_v2 = val_dataset.map(classes_to_ids).shuffle(seed=30)

# 在model的embedding层加入噪音进行数据增强

class BertForBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
        return {"loss": loss, "logits": logits, "probs": probs}


def model_init():
    model = BertForBinaryClassification.from_pretrained(f"{input_model}",
                                                        output_attentions=False,
                                                        output_hidden_states=False)
    model.resize_token_embeddings(len(tokenizer))
    return model

trainer = Trainer(
    model_init=model_init,
    data_collator=DataCollatorForCellClassification(),
    eval_dataset=val_dataset_v2
)


# 预测并保存结果
predictions = trainer.predict(val_dataset_v2)
probs = F.softmax(torch.tensor(predictions.predictions), dim=0).numpy()

fpr, tpr, thresholds = roc_curve(predictions.label_ids,probs[0, :, :])
auc = roc_auc_score(predictions.label_ids,probs[0,:,:])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC for test set')
plt.legend(loc="lower right")

plt.savefig(f'{output_path}/ROC_curve_test_set.pdf')
