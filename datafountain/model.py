from transformers import BertTokenizer, BertConfig
import transformers
import config
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Flatten
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_bert_model(path = config.MODEL_PATH):
    tokenizer = BertTokenizer.from_pretrained(path)

    model_config = BertConfig.from_pretrained(path+'bert_config.json')

    model_config.output_hidden_states = True
    model_config.output_attentions = True

    return tokenizer,model_config

tokenizer,model_config = get_bert_model()

class mymodel(transformers.BertPreTrainedModel):
    def __init__(self,conf,verbose=False,num_class=config.NUM_CLASS):
        super(mymodel,self).__init__(conf)
        self.bert_model = transformers.BertModel.from_pretrained(config.MODEL_NAME,config = conf)
        self.flatten = Flatten()
        self.ddense = Dense(1024,activation='relu')
        self.drop_out = Dropout(config.DROP_OUT)
        self.classifier = Dense(num_class,activation='sigmoid') #bert-large

    def call(self, ids, mask, token_type_ids):
        out = self.bert_model(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        #last hidden layer
        out = out[0]

        x = self.flatten(out)
        x = self.drop_out(x)
        x = self.ddense(x)
        logits = self.classifier(x)

        return logits
