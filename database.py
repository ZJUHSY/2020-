import os
import json
import re
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from bert_serving.client import BertClient
from random import sample

CLIENT_BATCH_SIZE = 4096
#SEN_NUM = 128
MIN_SEN_LEN = 5

# from bert_serving.server.helper import get_args_parser
# from bert_serving.server import BertServer
# bert_server_args = get_args_parser().parse_args(['-model_dir', 'model_dir chinese_L-12_H-768_A-12',
#                                      '-max_seq_len', '128'])
# server = BertServer(bert_server_args)
# server.start()

#cut paragraph to sentences
def cut_para(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.strip()  # remove both sizes' blanks
    return [sen.strip() for sen in para.split("\n") if len(sen.strip()) >= MIN_SEN_LEN]

'''
begin/end use as label to divide paras@@
'''

#customized loading data
class CustomDataset(Dataset):
    def __init__(self, path, sen_num, begin=0, end=None, is_train_data=True):
        self.is_train_data = is_train_data
        self.path = path
        if end == None and os.path.isfile(path + ".end"):
            self.data = torch.load(path + ".dat")
            self.label = torch.load(path + ".lab")
            self.start = torch.load(path + ".sta")
            self.end = torch.load(path + ".end")
            self.sen_num = sen_num
            return

        self.sen_num = sen_num
        #read data
        inp = open(path, "rb")
        passages = json.load(inp)
        self.passages = passages
        if is_train_data:
            self.cand_idx = sample(range(len(passages)), 100)
            passages = [passages[i] for i in range(len(passages)) if i not in self.cand_idx]





        sens = []
        self.label = []
        self.start = []
        self.end = []
        #pos_num, neg_num = 0, 0
        #pos_index = []
        #neg_index = []
        for passage in passages[begin: end]:
            pass_sen = cut_para(passage["passage"])
            if len(pass_sen) == 0:
                pass_sen = ['x']
            if len(pass_sen) > 2 * sen_num:
                pass_sen = pass_sen[0 : 2 * sen_num]
            self.start += [len(sens)]
            sens += pass_sen
            self.end += [len(sens)]
            if end == None:
                self.label += [passage["label"]]
            else:
                self.label += [0]
            # print(len(pass_sen))
        inp.close()
        # print('begin encoding')
        with BertClient() as be:
            self.data = be.encode(sens) #every senetcne an encoding vector
            self.data = torch.FloatTensor(self.data)
            torch.save(self.data, path + ".dat")
        torch.save(self.label, path + ".lab")
        torch.save(self.start, path + ".sta")
        torch.save(self.end, path + ".end")

        '''
            if len(pass_sen) < SEN_NUM:
                pass_sen += ["x"] * (SEN_NUM - len(pass_sen))
            for i in range(SEN_NUM):
                if not pass_sen[i]:
                    pass_sen[i] = "x"
            pass_sen = pass_sen[0: SEN_NUM]
            sens += pass_sen
            
            if 'label' in passage.keys():
                if passage["label"] == 1:
                    pos_num += 1
                    pos_index += [len(self.label)]
                    self.label += [1]
                else:
                    neg_num += 1
                    neg_index += [len(self.label)]
                    self.label += [0]
            else:
                neg_num += 1
                neg_index += [len(self.label)]
                self.label += [0]
        '''


        '''
        #send sentences to BERT-as-service, get each sentences' vector of size 768
        if balance:
            self.data = np.empty((len(sens) + abs(neg_num - pos_num) * SEN_NUM, 768), dtype=np.float32)
        else:
            self.data = np.empty((len(sens), 768), dtype=np.float32)
        last_num = 0
        while len(sens) > last_num:
            start = last_num
            end = min(last_num + CLIENT_BATCH_SIZE, len(sens))
            self.data[start : end] = be.encode(sens[start : end])
            last_num = end
            print("%s got %d/%d" % (path, last_num, len(sens)))
        #reshape the data for every passage
        self.data = np.resize(self.data, ((len(self.data) // SEN_NUM), SEN_NUM, 768))
       
        #balance the data
        last_num = last_num // SEN_NUM
        if balance:
            while pos_num < neg_num:
                self.data[last_num] = np.copy(self.data[pos_index[random.randint(0, len(pos_index) - 1)]])
                self.label.append(1)
                pos_num += 1
                last_num += 1

            while pos_num > neg_num:
                self.data[last_num] = np.copy(self.data[neg_index[random.randint(0, len(neg_index) - 1)]])
                self.label.append(0)
                neg_num += 1
                last_num += 1
        '''
    #get candidate database for train data
    def get_cand(self):
        if not self.is_train_data:
            return None
        if os.path.isfile(self.path + "_cand.lab"):
            data = torch.load(self.path + "_cand.dat")
            lens = torch.load(self.path + "_cand.lens")
            label = torch.load(self.path + "_cand.lab")
            return data, lens, label

        cand_passages = [self.passages[i] for i in range(len(self.passages)) if i in self.cand_idx]
        outp = open("train_cand.json", 'w', encoding="utf-8")
        outp.write(json.dumps(cand_passages, indent=4, ensure_ascii=False))
        outp.close()

        sens = []
        label = []
        lens = []
        for cand_passage in cand_passages:
            pass_sen = cut_para(cand_passage["passage"])
            if len(pass_sen) == 0:
                pass_sen = ['x']
            if len(pass_sen) > self.sen_num:
                start = random.randint(0, len(pass_sen)-self.sen_num)
                end = start + self.sen_num
                pass_sen = pass_sen[start:end]

            sens += pass_sen
            label += [cand_passage["label"]]
            lens += [len(pass_sen)]
        be = BertClient()
        data = be.encode(sens)
        torch.save(data, self.path + "_cand.dat")
        torch.save(lens, self.path + "_cand.lens")
        torch.save(label, self.path + "_cand.lab")
        return data, lens, label


    def __getitem__(self, index):
        SEN_NUM = self.sen_num
        if self.end[index] - self.start[index] <= SEN_NUM:
            para = self.data[self.start[index]: self.end[index]]
            length = self.end[index] - self.start[index]
            para = torch.cat((para, torch.zeros((SEN_NUM - (self.end[index] - self.start[index]), 768))), dim=0)
            # print("index padding: ", para.shape)
        else:
            start = random.randint(self.start[index], self.end[index] - SEN_NUM)
            #print(start)
            end = start + SEN_NUM
            length = SEN_NUM
            para = self.data[start : end]

        return para, length, self.label[index]

    def __len__(self):
        return len(self.label)