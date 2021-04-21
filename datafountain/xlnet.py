from transformers import XLNetTokenizer, XLNetConfig
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
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    # b. 导入配置文件
    model_config = XLNetConfig.from_pretrained(model_name)
    # 修改配置

    return tokenizer, model_config


tokenizer, model_config = get_bert_model()


class HealthModel(transformers.XLNetPreTrainedModel):
    def __init__(self, conf,
                 use_cuda=True, verbose=False, num_class=config.NUM_CLASS):
        super(HealthModel, self).__init__(conf)
        self.pre_model = transformers.XLNetModel.from_pretrained(config.MODEL_PATH, config=conf)
        self.use_cuda = use_cuda
        self.drop_out = nn.Dropout(config.DROP_OUT)
        self.bn = nn.BatchNorm1d(1024)
        self.dense = nn.Linear(1024*3,1024)
        self.fc = nn.Linear(768*4, num_class)

    def forward(self, ids,mask,seg_ids):
        out = self.pre_model(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = seg_ids
        )[0]
        a = torch.max(out,dim = 1)[0]
        b = torch.mean(out,dim = 1)
        c = out[:,-1]
        d = out[:,0]
        
        x = torch.cat([a,b,c,d],dim = 1)

        x = self.drop_out(x)
        logits = self.fc(x)

        return logits