import numpy as np
import config
# import tensorflow as tf
import data_loader
import model as premodel
import pandas as pd
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score
import time
import os
import transformers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from apex import amp



# from tensorflow.keras.optimizers import Adam
DEVICE = torch.device("cuda")

#load data
TRAIN_PATH = './train/hardtrain6.csv'
TEST_PATH = './test/test2.csv'

df_train = pd.read_csv(TRAIN_PATH,sep=' ')
df_test = pd.read_csv(TEST_PATH,sep=' ')


# df_train = shuffle(df_train,random_state = 2020)

# train_data,val_data = train_test_split(df_train,test_size = 0.1,shuffle=True,random_state=18)

# train_loader = data_loader.get_dataloader(train_data)
# val_loader = data_loader.get_dataloader(val_data)
# test_data = data_loader.get_dataloader(df_test)


#define loss and optimizer


def balance_bce(y_pred,y_gt):
    N = y_gt.size()[0]
    y_pred = y_pred.squeeze(1)

    loss = 0
    for pred,gt in zip(y_pred,y_gt):
        # a = -gt*torch.log(pred+1e-7)
        # b = -(1-gt)*torch.log(1-pred+1e-7)
        t_1 = (1-pred)**2
        t_2 = pred**2
        a = -gt*config.alpha*t_1*torch.log(pred+1e-8)
        b = -(1-gt)*(1-config.alpha)*t_2*torch.log(1-pred+1e-8)
        temp = a+b
        loss +=temp

    return loss/N

class FocalLossV1(nn.Module):
 
    def __init__(self,
                 alpha=config.alpha,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # self.crit = nn.BCEWithLogitsLoss(reduction='none')
 
        # self.celoss = torch.nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''
 
        # compute loss
        # logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        ce_loss=(-(label * torch.log(logits)) - (
                    (1 - label) * torch.log(1 - logits)))
        # ce_loss=(-(label * torch.log(torch.softmax(logits, dim=1))) - (
        #             (1 - label) * torch.log(1 - torch.softmax(logits, dim=1))))
        pt = torch.where(label == 1, logits, 1 - logits)
        # ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


balance_bce = nn.BCELoss()
# balance_bce = FocalLossV1()


#define Adversarial Training
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



def f1_accuracy(y_p, y_true):
    #
    thres = 0.5
    # y_p = y_p.squeeze(1)

    # for i in range(len(y_p)):
    #     if y_p[i] > 0.499:
    #         y_p[i] = 1.0
    #     else:
    #         y_p[i] = 0.0

    y_p = torch.round(y_p)
    y_p = y_p.int().cpu().numpy()
    y_true = y_true.int().cpu().numpy()
    return f1_score(y_true,y_p),thres

def search_f1(y_p, y_true):
    best = 0
    best_t = 0
    # y_true = y_true.int().cpu().numpy()
    for i in range(300,600):
        tres = i / 1000
        y_pred_bin =  (y_p > tres)#.int()#.cpu().numpy()
        score = f1_score(y_true, y_pred_bin.astype(int))
        if score > best:
            best = score
            best_t = tres
    print('thres', best_t)
    return best, best_t


def eval_step(model,val_loader):
    model.eval()
    y_true = torch.Tensor().float().to(DEVICE)
    y_p = torch.Tensor().float().to(DEVICE)
    total_loss = []
    with torch.no_grad():
        for dev_step, b in enumerate(iter(val_loader)):
            ids_1,mask_1,seg_ids,label = b['ids_1'].to(DEVICE),b['mask_1'].to(DEVICE),b['seg_ids'].to(DEVICE), b['label'].to(DEVICE)
            logits = model(ids_1,mask_1,seg_ids)
            logits = torch.sigmoid(logits).squeeze(1)
            label = label.float()
            y_pred = logits
            y_true = torch.cat((y_true, label),dim=0)
            y_p = torch.cat((y_p, y_pred),dim=0)
            loss = balance_bce(logits,label)
            total_loss.append(loss.item())
        cur_loss = np.mean(total_loss)
        # y_p = y_p.cpu().numpy()
        # y_true = y_true.cpu().numpy()
        # cur_acc ,thres= search_f1(y_p,y_true)
        cur_acc,thres = f1_accuracy(y_p,y_true)
        print('验证集损失：{:.3f},验证集f1值：{:<8.8f},阈值：{:.2f}'.format(cur_loss, cur_acc,thres))
    
        # if best_acc < cur_acc:
        #     best_acc = cur_acc
        #     if cur_acc > save_acc:
        #         print("Save best model, accuary={}".format(best_acc))
        #         save_model(model_dir = './xlnet_save/', model_name = 'xlnet_421e-5_0.3_lr_adv_2accumulation{:.4f}thres{:.4f}'.format(cur_acc,thres), model = model)
    return y_p.cpu().numpy()

def pred_step(model,test_loader):
    model.eval()
    y_p = torch.Tensor().float().to(DEVICE)
    with torch.no_grad():
        for dev_step, b in enumerate(iter(test_loader)):
            ids,masks,type_ids = b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE),b['token_type_ids'].to(DEVICE)
            logits = model(ids = ids, mask = masks, seg_ids = type_ids)
            logits = torch.sigmoid(logits).squeeze(1)
            y_p = torch.cat([y_p,logits],dim = 0)

    return y_p.cpu().numpy()




def save_model(model_dir, model_name, model):     
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    print("Saved model to", model_dir)
    torch.save(model.state_dict(), model_path)


    


save_acc = config.SAVE_ACC



gkf = GroupKFold(n_splits=5).split(X=df_train.reply, groups=df_train.ID)
oof = np.zeros(len(df_train))

prediction = np.zeros(len(df_test))

test_loader = data_loader.get_dataloader(df_test,'test')



for fold,(tra_id,val_id) in enumerate(gkf):
    print("NO.{}".format(fold+1))
    
    train_loader = data_loader.get_dataloader(df_train.iloc[tra_id,:],'train')
    val_loader = data_loader.get_dataloader(df_train.iloc[val_id,:],'val')
    #load model
    model = premodel.HealthModel(conf=premodel.model_config)
    # model = nn.DataParallel(model)
    if config.USE_CUDA:
        model.to(DEVICE)
    
    
    #define optimizer
    # param_optimizer = list(model.bert_model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    #     {'params':model.fc.parameters(),'lr':2e-5},
    #     {'params':model.dense.parameters(),'lr':2e-5}
    # ]

    optimizer = torch.optim.Adam([{'params':model.pre_model.parameters(), 'lr':config.LEARNING_RATE},{'params':model.fc.parameters(),'lr':1e-5}],lr = 3e-5)
    #[{'params':model.bert_model.parameters(), 'lr':config.LEARNING_RATE},{'params':model.fc.parameters(),'lr':1e-5}]
    #model.parameters()
    # lr_schedule = transformers.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = 60,num_training_steps = int(len(train_data)/config.TRAIN_BATCH_SIZE/config.ACCUMULATION_STEPS))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4,eta_min=1e-5)
    fgm = FGM(model)

    if config.fp16:
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
#train and val
    for epoch in range(config.EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, config.EPOCHS))
        model.train()
        for step, batch in tqdm.tqdm(enumerate(iter(train_loader))):
            if step == 0: start_time = time.time()
            ids_1,mask_1,seg_ids,label = batch['ids_1'].to(DEVICE),batch['mask_1'].to(DEVICE),batch['seg_ids'].to(DEVICE),batch['label'].to(DEVICE)
            logits = model(ids_1,mask_1,seg_ids)
            logits = torch.sigmoid(logits)
            logits = logits.squeeze(1)
            label = label.float()
        

            loss = balance_bce(logits,label)
            loss = loss/config.ACCUMULATION_STEPS
            if config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            #adversarial training
            fgm.attack()
            logits_adv = model(ids_1,mask_1,seg_ids)
            logits_adv = torch.sigmoid(logits_adv).squeeze(1)
            loss_adv = balance_bce(logits_adv,label)
            loss_adv = loss_adv/config.ACCUMULATION_STEPS
            if config.fp16:
                with amp.scale_loss(loss_adv, optimizer) as scaled_loss_adv:
                    scaled_loss_adv.backward()
            else:
                loss_adv.backward()
            fgm.restore()

            #clip_grad
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            #optimize the net
            if ((step+1)%config.ACCUMULATION_STEPS )==0: 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            
            if (step+1) % 10 == 0:
                end_time = time.time()
                print('epoch: %d, step: %d/%d, loss: %.5f, time_per_step: %.3f' % (
                    epoch+1, step, len(train_loader), loss, (end_time - start_time)/(step+1) ))

    oof[val_id] = eval_step(model,val_loader)
    prediction += pred_step(model,test_loader) 

thres = 0.5

score = f1_score(df_train['label'].values,np.around(oof))
print(score)
oofdf = pd.DataFrame()
oofdf['label'] = oof
oofdf.to_csv('./oof_save/5m_hardps_bertlarge_cv_431e-5_0.5_lr_adv_4accumulation{:.4f}thres{:.4f}.csv'.format(score,thres),index=False,header=None,sep='\t')
save_model(model_dir = './model6_save/', model_name = '5m_hardps_bertlarge_cv_431e-5_0.5_lr_adv_4accumulation{:.4f}thres{:.4f}'.format(score,thres), model = model)

sub = prediction /5
# sub = np.around(sub)


submit = pd.DataFrame()
submit['label'] = sub#.astype(int)
submit.to_csv('./oof_test/5m_hardps_bertlarge_cv_431e-5_0.5_lr_adv_4accumulation{:.4f}thres{:.4f}.csv'.format(score,thres),index = False,header = None)#,sep = '\t')
submit['ID'] = df_test.ID
submit['RID'] = df_test.RID
sub = np.around(sub)
submit['label'] = sub.astype(int)
submit.to_csv('./sub_save/5m_hardps_bertlarge_cv_431e-5_0.5_lr_adv_4accumulation{:.4f}thres{:.4f}.csv'.format(score,thres),index = False,header = None,sep = '\t')
print("save submit")





