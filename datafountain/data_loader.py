import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import config
from model import tokenizer
AUTOTUNE = tf.data.experimental.AUTOTUNE

class HealthDataset:
    def __init__(self, question, question_id, reply_id, label):
        self.question = question
        self.question_id = question_id
        self.reply_id = reply_id
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        process_data = data_process(
            self.question[item],
            self.label[item],
            self.tokenizer,
            self.max_len
        )
        ids = np.array(process_data['input_ids'])
        masks = np.array(process_data['attention_mask'])
        type_ids = np.array(process_data['token_type_ids'])

        return {
            'question_id': self.question_id[item],
            'reply_id': self.reply_id[item],
            'input_ids': ids,
            'attention_mask': masks,
            'token_type_ids': type_ids,
            'label': np.array(process_data['label'])
        }


def data_process(question, label, tokenizer, max_len):
    question_code = tokenizer.encode_plus(question, max_length=max_len, pad_to_max_length=True, truncation=True)
    ids = question_code['input_ids']
    mask = question_code['attention_mask']
    token_type = question_code['token_type_ids']
    label = label

    return {
        'input_ids': ids,
        'attention_mask': mask,
        'token_type_ids': token_type,
        'label': label
    }


def get_dataloader(df_train):
    train_dataset = HealthDataset(
        question=df_train['question_reply'].values,
        question_id=df_train['ID'].values,
        reply_id=df_train['RID'].values,
        label=df_train['label'].values
    )
    train_dataset = pd.Dataframe(train_dataset)
    train_dataloader = tf.data.experimental.make_csv_dataset(train_dataset,batch_size=config.TRAIN_BATCH_SIZE)
    train_dataloader = train_dataloader.prefetch(buffer_size=AUTOTUNE)

    return train_dataloader