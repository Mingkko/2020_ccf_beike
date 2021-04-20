import numpy as np
import pandas as pd
import re

# PATH = './train/'
#
# train_query = pd.read_csv(PATH+'train.query.tsv',sep = '\t',header=None)
#
#
# train_query.rename(columns={0:'ID',1:'question'},inplace=True)
# print(train_query.head())
#
# train_reply = pd.read_csv(PATH+'train.reply.tsv',sep='\t',header=None)
#
# train_reply.rename(columns={0:'ID',1:'RID',2:'reply',3:'label'},inplace=True)
# def filter_emoji(content):
#     try:
#         # Wide UCS-4 build
#         cont = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
#     except re.error:
#         # Narrow UCS-2 build
#         cont = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
#     return cont.sub (u'', content)
#
# train_data = pd.DataFrame()
# train_data['ID'] = train_reply.ID
# train_data['RID'] = train_reply.RID
# train_data['label'] = train_reply.label
# train_data['reply'] = train_reply.reply
#
# print(train_data.columns)
# for i in range(train_data.shape[0]):
#     temp = train_data.iloc[i,0]
#     train_data.loc[i,'question'] = train_query[train_query['ID']==temp]['question'].values
#     # print()
#
#
#
# train_data = train_data.dropna()
# train_data['question'] = train_data['question'].apply(filter_emoji)
# train_data['reply'] = train_data['reply'].apply(filter_emoji)
# train_data.to_csv(PATH+'train2.csv',index=False,sep=' ')
# print(train_data.head())
#
# PATH2 = './test/'
# test_query = pd.read_csv(PATH2+'test.query.tsv',sep='\t',encoding='gbk',header=None)
# test_reply = pd.read_csv(PATH2+'test.reply.tsv',sep='\t',encoding='gbk',header=None)
#
# test_query.rename(columns={0:'ID',1:'question'},inplace=True)
# test_reply.rename(columns={0:'ID',1:'RID',2:'reply'},inplace=True)
#
# print(test_query.head())
# print(test_reply.head())
# # exit(0)
#
# test_data = pd.DataFrame()
# test_data['ID'] = test_reply.ID
# test_data['RID'] = test_reply.RID
#
# test_data['reply'] = test_reply.reply
#
# for i in range(test_data.shape[0]):
#     temp = test_data.iloc[i,0]
#     test_data.loc[i,'question'] = test_query[test_query['ID']==temp]['question'].values
#
#
#
# print(test_data.head().append(test_data.tail()))
# test_data['question'] = test_data['question'].apply(filter_emoji)
# test_data['reply'] = test_data['reply'].apply(filter_emoji)
# test_data.to_csv(PATH2+'test2.csv',index=False,sep=' ')

train_left = pd.read_csv('./train/train.query.tsv',sep='\t',header=None)
train_left.columns=['ID','question']
train_right = pd.read_csv('./train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['ID','RID','reply','label']
df_train = train_left.merge(train_right, how='left')
df_train['reply'] = df_train['reply'].fillna('好的')
test_left = pd.read_csv('./test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['ID','question']
test_right =  pd.read_csv('./test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['ID','RID','reply']
df_test = test_left.merge(test_right, how='left')

df_train.to_csv('./train/train3.csv',index = False,sep = ' ')
df_test.to_csv('./test/test3.csv',index = False,sep=' ')