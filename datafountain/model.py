from transformers import BertTokenizer, BertConfig
import transformers
import config
# import tensorflow as tf
# from tensorflow.keras.layers import Dense,Dropout,Flatten
# AUTOTUNE = tf.data.experimental.AUTOTUNE
import torch.nn as nn
import torch
import numpy as np

def get_bert_model(model_name=config.MODEL_NAME):
    # a. 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained(model_name+'google_zh_vocab.txt')
    # b. 导入配置文件
    model_config = BertConfig.from_pretrained(model_name)
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    return tokenizer, model_config


tokenizer, model_config = get_bert_model()


class HealthModel(transformers.BertPreTrainedModel):
    def __init__(self, conf,
                 use_cuda=True, verbose=False, num_class=config.NUM_CLASS):
        super(HealthModel, self).__init__(conf)
        self.pre_model = transformers.BertModel.from_pretrained(config.MODEL_PATH, config=conf)
        self.use_cuda = use_cuda
        self.drop_out = nn.Dropout(config.DROP_OUT)
        # self.bn = nn.BatchNorm1d(1024)
        # self.dense = nn.Linear(1024*3,1024)
        # self.fc = nn.Linear(768,num_class)
        self.fc = nn.Linear(1024, num_class)

    def forward(self, ids,mask,seg_ids):
        out = self.pre_model(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = seg_ids
        )
        x = out[0][:,0,:]
        # sequence2 = out[2][-2][:,0,:]
        # pool = out[1]

        # # sequence1 = torch.mean(sequence1,dim = 1)

        # concat = torch.cat([sequence1,pool],dim = 1)
        
        # x = self.dense(concat)
        # x = self.bn(x)
        # x = nn.functional.relu(x)

        x = self.drop_out(x)
        logits = self.fc(x)

        return logits