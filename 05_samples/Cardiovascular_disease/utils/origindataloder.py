import collections
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset 
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F

def ap_hilo_feat(x, y):
    #至適血圧
    if x < 120 and y < 80:
        return  torch.eye(7)[0]
    #正常血圧 
    if (120 <= x and x < 130) or (80 <= y and y < 85):
        return  torch.eye(7)[1]
    #正常高値 
    if (129 < x and x < 140) or (84 < y and y < 90):
        return  torch.eye(7)[2]
    #Ⅰ度高血圧
    if (139 < x and x < 160) or (89 < y and y < 100):
        return  torch.eye(7)[3]
    #Ⅱ度高血圧
    if (159 < x and x < 180) or (99 < y and y < 110):
        return  torch.eye(7)[4]
    #Ⅲ度高血圧
    if 180 <= x or 110 <= y:
        return  torch.eye(7)[5]
    #(孤立性)収縮期高血圧
    if (140 <= x) and (y <= 90):
        return  torch.eye(7)[6]


def bmi_feature(h, w):
    bmi = w / (h * h)
    #低体重
    if bmi < 18.5:
        return  torch.eye(6)[0]
    #標準体重
    if 18.5 <= bmi or bmi < 25:
        return  torch.eye(6)[1]
    #肥満度１
    if 25 <= bmi or bmi < 30:
        return  torch.eye(6)[2]
    #肥満度２
    if 30 <= bmi or bmi < 35:
        return  torch.eye(6)[3]
    #肥満度３
    if 35 <= bmi or bmi < 40:
        return  torch.eye(6)[4]
    #肥満度4
    if 40 <= bmi:
        return  torch.eye(6)[5]

def age_feature_tensor(age):
    if age <= 30 * 365:
        return  torch.eye(4)[0]
    if age  <= 40 * 365:
        return  torch.eye(4)[1]
    if age < 65 * 365:
        return  torch.eye(4)[2]
    if age >= 65 * 365:
        return  torch.eye(4)[3]



class MyDatasets(Dataset):

    def __init__(self, df):
        self.len = len(df.values)
        self.age = df[["age"]].values
        self.gender = df[["gender"]].values
        self.height = df[["height"]].values
        self.weight = df[["weight"]].values
        self.aphi = df[["ap_hi",]].values
        self.aplo = df[["ap_lo"]].values
        self.chol = df[["cholesterol"]].values
        self.gluc = df[["gluc"]].values
        self.smoke = df[["smoke"]].values
        self.alco = df[["alco"]].values
        self.active= df[["active"]].values
        self.label = df[["cardio"]].values

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #年齢 1~4
        #age = int(self.age[idx] / 3650)
        age_feature = age_feature_tensor(self.age[idx]) # 11
        #BMI　5~9
        bmi = bmi_feature(self.height[idx] / 100, self.weight[idx])
        #gender 10,11
        gen = self.gender[idx]
        gen_feature = torch.eye(2)[gen -1]
        #血圧評価　12~18
        ap_hilo = ap_hilo_feat(self.aphi[idx], self.aplo[idx])
        #コレステロール　19~21
        chol_feature = torch.eye(3)[self.chol[idx] -1]
        #グルコース　22~24
        gluc_feature = torch.eye(3)[self.gluc[idx] -1]
        #喫煙　25
        smoke_feature = torch.tensor(self.smoke[idx], dtype=torch.float32)
        #飲酒  26
        alco_feature = torch.tensor(self.alco[idx], dtype=torch.float32)
        #身体活動  27
        active_feature = torch.tensor(self.active[idx], dtype=torch.float32)

        x = torch.cat([age_feature, bmi, gen_feature.view(-1), ap_hilo, chol_feature.view(-1), gluc_feature.view(-1), smoke_feature, alco_feature, active_feature], dim=0)
        
        y = torch.eye(2)[self.label[idx]].view(-1)
       
        return  x, y

class MyDatasets2(Dataset):

    def __init__(self, df):
        self.len = len(df.values)
        self.age = df[["age"]].values
        self.gender = df[["gender"]].values
        self.height = df[["height"]].values
        self.weight = df[["weight"]].values
        self.aphi = df[["ap_hi",]].values
        self.aplo = df[["ap_lo"]].values
        self.chol = df[["cholesterol"]].values
        self.gluc = df[["gluc"]].values
        self.smoke = df[["smoke"]].values
        self.alco = df[["alco"]].values
        self.active= df[["active"]].values
        self.label = df[["cardio"]].values

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #年齢 11
        #age = int(self.age[idx] / 3650)
        age_feature = age_feature_tensor2(self.age[idx]) # 11
        #BMI　6
        bmi = bmi_feature2(self.height[idx] / 100, self.weight[idx])
        #gender 1
        gen = self.gender[idx]
        gen_feature = torch.eye(2)[gen -1]
        #血圧評価　6
        ap_hilo = ap_hilo_feat2(self.aphi[idx], self.aplo[idx])
        #コレステロール　3
        chol_feature = torch.eye(3)[self.chol[idx] -1]
        #グルコース　3
        gluc_feature = torch.eye(3)[self.gluc[idx] -1]
        #喫煙　1
        smoke_feature = torch.tensor(self.smoke[idx], dtype=torch.float32)
        #飲酒  1
        alco_feature = torch.tensor(self.alco[idx], dtype=torch.float32)
        #身体活動  1
        active_feature = torch.tensor(self.active[idx], dtype=torch.float32)

        x = torch.cat([age_feature, bmi, gen_feature.view(-1), ap_hilo, chol_feature.view(-1), gluc_feature.view(-1), smoke_feature, alco_feature, active_feature], dim=0)
        
        y = torch.eye(2)[self.label[idx]].view(-1)
        #y = torch.tensor(self.label[idx], dtype=torch.int64)
        return  x, y





def ap_hilo_feat2( x, y):
    #至適血圧
    if x < 120 and y < 80:
        return F.embedding(torch.tensor(0, dtype=torch.int64), torch.rand(6,6))
    #正常血圧 
    if (120 <= x and x < 130) or (80 <= y and y < 85):
        return F.embedding(torch.tensor(1, dtype=torch.int64), torch.rand(6,6))
    #正常高値 
    if (129 < x and x < 140) or (84 < y and y < 90):
        return F.embedding(torch.tensor(2, dtype=torch.int64), torch.rand(6,6))
    #Ⅰ度高血圧
    if (139 < x and x < 160) or (89 < y and y < 100):
        return F.embedding(torch.tensor(3, dtype=torch.int64), torch.rand(6,6))
    #Ⅱ度高血圧
    if (159 < x and x < 180) or (99 < y and y < 110):
        return F.embedding(torch.tensor(4, dtype=torch.int64), torch.rand(6,6))
    #Ⅲ度高血圧
    if 180 <= x or 110 <= y:
        return F.embedding(torch.tensor(5, dtype=torch.int64), torch.rand(6,6))
    #(孤立性)収縮期高血圧
    if (140 <= x) and (y <= 90):
        return F.embedding(torch.tensor(6, dtype=torch.int64), torch.rand(6,6))

def bmi_feature2(h, w):
    bmi = w / (h * h)
    #低体重
    if bmi < 18.5:
        return F.embedding(torch.tensor(0, dtype=torch.int64), torch.rand(5,6))
    #標準体重
    if 18.5 <= bmi or bmi < 25:
        return F.embedding(torch.tensor(1, dtype=torch.int64), torch.rand(5,6))
    #肥満度１
    if 25 <= bmi or bmi < 30:
        return F.embedding(torch.tensor(2, dtype=torch.int64), torch.rand(5,6))
    #肥満度２
    if 30 <= bmi or bmi < 35:
        return F.embedding(torch.tensor(3, dtype=torch.int64), torch.rand(5,6))
    #肥満度３
    if 35 <= bmi or bmi < 40:
        return F.embedding(torch.tensor(4, dtype=torch.int64), torch.rand(5,6))
    #肥満度4
    if 40 <= bmi:
        return F.embedding(torch.tensor(5, dtype=torch.int64), torch.rand(5,6))

def age_feature_tensor2(age):
    if age <= 20 * 365:
        return F.embedding(torch.tensor(0, dtype=torch.int64), torch.rand(6,10))
    if age  <= 30 * 365:
        return F.embedding(torch.tensor(1, dtype=torch.int64), torch.rand(6,10))
    if age <= 40 * 365:
        return F.embedding(torch.tensor(2, dtype=torch.int64), torch.rand(6,10))
    if age <= 50 * 365:
        return F.embedding(torch.tensor(3, dtype=torch.int64), torch.rand(6,10))
    if age <= 60 * 365:
        return F.embedding(torch.tensor(4, dtype=torch.int64), torch.rand(6,10))
    if age >= 60 * 365:
        return F.embedding(torch.tensor(5, dtype=torch.int64), torch.rand(6,10))