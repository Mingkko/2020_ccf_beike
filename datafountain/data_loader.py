import numpy as np
import config
from model import tokenizer
import torch

class HealthDataset:
    def __init__(self, question, reply,question_id, reply_id, label=None,type = 'train'):
        self.question = question
        self.reply = reply
        self.question_id = question_id
        self.reply_id = reply_id
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = config.MAX_LEN
        self.type = type

    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        if self.type == 'train':
            process_data = data_process(
                self.question[item],
                self.reply[item],
                self.tokenizer,
                self.max_len
            )
            ids_1 = torch.tensor(process_data['ids_1'],dtype = torch.long)
            mask_1 = torch.tensor(process_data['mask_1'],dtype = torch.long)
            token_type = torch.tensor(process_data['seg_ids'],dtype = torch.long)

            return {
                'question_id':self.question_id[item],
                'reply_id':self.reply_id[item],
                'ids_1':ids_1,
                'mask_1':mask_1,
                'seg_ids':token_type,
                'label': torch.tensor(self.label[item], dtype = torch.long)
            }
        else:
            process_data = data_process(
                self.question[item],
                self.reply[item],
                self.tokenizer,
                self.max_len
            )
            ids = torch.tensor(process_data['ids_1'], dtype=torch.long)
            masks = torch.tensor(process_data['mask_1'], dtype=torch.long)
            type_ids = torch.tensor(process_data['seg_ids'], dtype=torch.long)
            return {
                'question_id':self.question_id[item],
                'reply_id':self.reply_id[item],
                'input_ids':ids,
                'attention_mask':masks,
                'token_type_ids':type_ids,
            }


def data_process(question, reply, tokenizer, max_len):

    setence_pair = tokenizer.encode_plus(question,reply,max_length=max_len, padding='max_length', truncation=True)
    ids_1 = setence_pair['input_ids']
    mask_1 = setence_pair['attention_mask']
    token_type = setence_pair['token_type_ids']


    return {
        'ids_1':ids_1,
        'mask_1':mask_1,
        'seg_ids':token_type,
    }


def get_dataloader(df_data,type = 'train'):
    
    if type == 'train' or type == 'val':
        dataset = HealthDataset(
            question=df_data['question'].values,
            reply = df_data['reply'].values,
            question_id=df_data['ID'].values,
            reply_id=df_data['RID'].values,
            label=df_data['label'].values
        )
    else:
        dataset = HealthDataset(
            question=df_data['question'].values,
            reply = df_data['reply'].values,
            question_id = df_data['ID'].values,
            reply_id = df_data['RID'].values,
            type = 'test'
        )

    if type == 'train':
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=True,  
            num_workers=2
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=False,  
            num_workers=2
        )

    return data_loader