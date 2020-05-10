import sys
import os
import argparse
import json
import numpy as np

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from database import CustomDataset  # , SEN_NUM
from model import relational_attention


# from model import CNN_Text, test

# BERT_MAX_SEQ_LEN = 64
# CHECK_TIME = 1

class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(768, args.hidden_layer, num_layers=2, bidirectional=True)
        # self.pooling = nn.MaxPool1d(args.sen_num)
        self.linear1 = nn.Linear(args.hidden_layer * 4, args.hidden_layer // 2)
        self.dropout = nn.Dropout(0.5)
        self.encoding_k_layer = nn.Linear(args.hidden_layer // 2, args.attention_dim)
        self.encoding_q_layer = nn.Linear(args.hidden_layer // 2, args.attention_dim)
        self.encoding_v_layer = nn.Linear(args.hidden_layer // 2, args.attention_dim)
        # self.encoding_v_layer = F.relu(self.dropout(nn.Linear(args.hidden_Layer // 2, args.attention_dim)))
        self.val_norm = nn.LayerNorm(args.attention_dim)

    def forward(self, sens, lens):
        # print(lens)
        lens, perm_idx = lens.sort(0, descending=True)
        # print(lens)
        # print(perm_idx)
        sens = sens[perm_idx]
        sens = sens.permute(1, 0, 2)  # B * L * V -> L * B * V
        # print('sens:{}'.format(sens.shape))
        sens = pack_padded_sequence(sens, lens, batch_first=False, enforce_sorted=True)
        o, (h, c) = self.lstm(sens)  # o: <L * B * 2V
        # print(type(o))
        # print('h:{}'.format(h.shape))
        # print('c:{}'.format(c.shape))
        o_max = pad_packed_sequence(o, batch_first=False, padding_value=float("-inf"), total_length=None)[0]  # L * B * 2V
        # print('o_max:{}'.format(o_max.shape))
        h_max = F.max_pool1d(o_max.permute(1, 2, 0), o_max.shape[0]).squeeze(2)  # B * 2V
        # print('h_max:{}'.format(h_max.shape))
        o_avg = pad_packed_sequence(o, batch_first=False, padding_value=0, total_length=None)[0]  # L * B * 2V
        # print('o_avg:{}'.format(o_avg.shape))
        h_avg = torch.div(torch.sum(o_avg, dim=0).permute(1, 0), lens.to(dtype=torch.float)).permute(1, 0)  # B * 2V

        h = torch.cat((h_max, h_avg), dim=1)  # B * 4V
        _, unperm_idx = perm_idx.sort(0)
        h = h[unperm_idx]

        # get K,Q,V
        dense = self.linear1(h)
        encoded_keys = self.encoding_k_layer(dense)
        encoded_keys = F.relu(self.dropout(encoded_keys))
        encoded_querys = self.encoding_q_layer(dense)
        encoded_querys = F.relu(self.dropout(encoded_querys))
        encoded_values = self.encoding_v_layer(dense)
        encoded_values = F.relu(self.dropout(encoded_values))
        encoded_values = self.val_norm(encoded_values)
        return encoded_keys, encoded_querys, encoded_values


class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model, self).__init__()
        self.linear1 = nn.Linear(args.attention_dim, 2)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f):
        logits = self.dropout(self.linear1(f))
        soft_pred = self.softmax(logits)
        return soft_pred  # self.softmax(f)


if __name__ == "__main__":
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-r", "--regularization", type=float, default=0.001)  # normally 0.0005

    # relatively loose hyperparameters
    parser.add_argument("-e", "--epoch", type=int, default=50)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-c", "--clip", type=float, default=1)
    parser.add_argument("-hl", "--hidden_layer", type=int, default=100)
    parser.add_argument("-de", "--decay_epoch", type=int, default=20)
    parser.add_argument("-ct", "--check_time", type=int, default=1)
    parser.add_argument("-sn", "--sen_num", type=int, default=200)
    parser.add_argument("-tm", "--train_model", type=bool, default=True)
    parser.add_argument("-s", "--step", type=int, default=256)
    parser.add_argument("-nc", "--num_classes", type=int, default=2)
    parser.add_argument("-ad", "--attention_dim", type=int, default=32)
    parser.add_argument("-cs", "--cand_size", type=int, default=100)
    parser.add_argument("-nom", "--normalization", type=str, default="softmax")
    parser.add_argument("-ep_spar", "--epsilon_sparsity", type=float, default=0.000001)
    parser.add_argument("-ai", "--alpha_intermediate", type=float, default=0.5)
    parser.add_argument("-sw", "--sparsity_weight", type=float, default=0.0001)

    args = parser.parse_args()

    # use CUDA to speed up
    # print(args.epoch, args.batch_size, args.train_model)
    use_cuda = torch.cuda.is_available()

    # get data
    train_dataset = CustomDataset(path="train.json", sen_num=args.sen_num)
    #cand data processing first (no batch input)
    cand_sens, cand_lens, cand_labels = train_dataset.get_cand()  # get candidate data cand
    cum_lens = [0] + list(np.cumsum(cand_lens))
    sens_list = [torch.FloatTensor(cand_sens[cum_lens[i]: cum_lens[i+1]]) for i in range(len(cand_lens))]
    for i, para in enumerate(sens_list):
        if para.shape[0] < args.sen_num:
            para = torch.cat((para, torch.zeros(args.sen_num - para.shape[0], 768)), dim=0)
        para = para.unsqueeze(0)
        sens_list[i] = para
            # print('para.shape: {}'.format(para.shape))
    # print(sens_list[0].shape)
    cand_sens = torch.cat(sens_list, dim=0)
    cand_lens = torch.IntTensor(cand_lens)
    cand_labels = torch.IntTensor(cand_labels)
    # print('cand_sens shape: {}'.format(cand_sens.shape))
    if use_cuda:
        cand_sens = cand_sens.cuda()
        cand_lens = cand_lens.cuda()
        cand_labels = cand_labels.cuda()


    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True)  # , collate_fn=collate_wrapper, pin_memory=True)
    dev_loader = Data.DataLoader(dataset=CustomDataset(path="dev.json", sen_num=args.sen_num,
                                                       is_train_data=False),
                                 batch_size=args.batch_size,
                                 shuffle=False)  # , collate_fn=collate_wrapper, pin_memory=True)

    # initialize model
    lstm = LSTM_model()
    mlp = MLP_model()
    if use_cuda:
        lstm = lstm.cuda()
        mlp = mlp.cuda()
        # relational_attention = relational_attention.cuda()
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(mlp.parameters()), lr=learning_rate,
                                 weight_decay=args.regularization)


    def run(data_loader, update_model, predict=False):
        total_num = 0
        right_num = 0
        true_pos = 0
        pos = 0
        true = 0
        total_loss = 0

        iteration = 1 if update_model else args.check_time

        for ite in range(iteration):
            for step, train_data in enumerate(data_loader):
                # print(step)
                train_sens, train_lens, train_labels = train_data
                # print('Sen_dim: {}'.format(sens.shape))
                # print('len:{}'.format(lens))
                if use_cuda:
                    train_sens = train_sens.cuda()
                    train_lens = train_lens.cuda()
                    train_labels = train_labels.cuda()
                _, encoded_batch_queries, encoded_batch_values = lstm(train_sens, train_lens)
                # print('train_query_shape: {}'.format(encoded_batch_queries.shape))
                # print('train_value_shape: {}'.format(encoded_batch_values.shape))

                encoded_cand_keys, _, encoded_cand_values = lstm(cand_sens, cand_lens)
                # print('Cand_keys_shape: {}'.format(encoded_cand_keys.shape))
                # print("cand_values_shape: {}".format(encoded_cand_values.shape))

                weighted_encoded_batch, weight_coefs_batch = relational_attention(encoded_batch_values.cpu(),
                                                                                  encoded_cand_keys.cpu(),
                                                                                  encoded_cand_values.cpu(),
                                                                                  normalization=args.normalization)

                # print("Weighted encoded batch shape: {}".format(weighted_encoded_batch.shape))
                # print("weight_coefs_bacth shape: {}".format(weight_coefs_batch.shape))

                if use_cuda:
                    weighted_encoded_batch = weighted_encoded_batch.cuda()
                    weight_coefs_batch = weight_coefs_batch.cuda()

                # Sparsity regularization
                # print("-------------------------batch loss:-------------------------------")
                tmp_weights = -weight_coefs_batch * torch.log(args.epsilon_sparsity + weight_coefs_batch)
                entropy_weights = tmp_weights.sum(1)
                sparsity_loss = torch.mean(entropy_weights) - torch.log(
                    torch.tensor(args.epsilon_sparsity + 100, dtype=float))
                # print("Sparsity_loss: {} ".format(sparsity_loss), end="")

                # Intermediate loss
                joint_encoded_batch = (1 - args.alpha_intermediate) * encoded_batch_values \
                                      + args.alpha_intermediate * weighted_encoded_batch

                joint_pred = mlp(joint_encoded_batch)
                joint_loss = F.cross_entropy(joint_pred, train_labels)
                # print("Joint_loss: {} ".format(joint_loss), end="")

                # self loss
                self_pred = mlp(encoded_batch_values)
                self_loss = F.cross_entropy(self_pred, train_labels)
                # print("Self_loss: {} ".format(self_loss), end="")

                # Prototype combination loss
                proto_pred = mlp(weighted_encoded_batch)
                proto_loss = F.cross_entropy(proto_pred, train_labels)
                # print("Proto_loss: {} ".format(proto_loss), end="")

                batch_loss = joint_loss + self_loss + proto_loss + args.sparsity_weight * sparsity_loss
                # print("Batch_loss: {} ".format(batch_loss), end="")


                if update_model:
                    total_loss += batch_loss
                    optimizer.zero_grad()
                    batch_loss.backward()
                    clip_grad_norm_(lstm.parameters(), args.clip)
                    optimizer.step()
                # print(lens[labels != torch.max(score, 1)[1]])
                if step == 0:
                    pred = torch.max(joint_pred, 1)[1]
                    val = joint_pred  # semantic value
                    targ = train_labels
                    cand_weight = weight_coefs_batch
                else:
                    pred = torch.cat((pred, torch.max(joint_pred, 1)[1]), dim=0)
                    targ = torch.cat((targ, train_labels), dim=0)
                    val = torch.cat((val, joint_pred), dim=0)
                    cand_weight = torch.cat((cand_weight, weight_coefs_batch), dim=0)
                    # print(val.shape)
                    # print(val)
                    # print(cand_weight.shape)
            if ite == 0:
                pred_sum = pred
                weight_sum = cand_weight
            else:
                pred_sum += pred
                weight_sum += cand_weight

        pred = torch.zeros(pred.shape).to(dtype=torch.int64)
        if use_cuda:
            pred = pred.cuda()
        pred[pred_sum > (iteration * 1.0 / 2)] = 1
        # val = pred_sum/iteration #semantic value
        labels = targ

        if predict:
            return pred.cpu(), val.cpu()

        right_num += labels[pred == labels].size(0)
        total_num += labels.size(0)
        true_pos += labels[(labels == 1) & (pred == labels)].size(0)
        pos += labels[pred == 1].size(0)
        true += labels[labels == 1].size(0)

        if update_model:
            print("train: loss: {} ".format(total_loss), end="")
        else:
            print("dev: ", end="")
        accuracy = float(right_num) / total_num

        print("accuracy: {} ".format(accuracy), end="")
        if pos > 0:
            precision = float(true_pos) / pos
            print("precision: {} ".format(precision), end="")
        if true > 0:
            recall = float(true_pos) / true
            print("recall: {} ".format(recall), end="")
        if pos > 0 and true > 0 and (precision + recall) > 0:
            F1 = 2.0 * precision * recall / (precision + recall)
            print("F1: {} ".format(F1))
            return F1, pred.cpu(), weight_sum.cpu()
        else:
            print()
        return None


    if args.train_model:  # train model
        F1_max = 0
        for epoch in range(args.epoch):
            print("epoch:{}".format(epoch + 1))
            run(data_loader=train_loader, update_model=True)
            F1, pred, cand_weight = run(data_loader=dev_loader, update_model=False)

            if F1 > F1_max:
                torch.save(lstm.state_dict(), "lstm.pk")
                torch.save(mlp.state_dict(), "mlp.pk")
                print("F1: {} saved".format(F1))
                F1_max = F1
                if not os.path.exists("Proto_VIS"):
                    os.mkdir("Proto_VIS")
                else:
                    torch.save(pred, ".\\Proto_VIS\\pred")
                    torch.save(cand_weight, '.\\Proto_VIS\\cand_weight')
                # outp = open("cand_weight_dev.json", 'w', encoding="utf-8")
                # outp.write(json.dumps(cand_weight, indent=4, ensure_ascii=False))
                # outp.close()
    else:  # run unlabled data
        lstm.load_state_dict(torch.load("lstm.pk"))
        lstm.eval()
        mlp.load_state_dict(torch.load("mlp.pk"))
        mlp.eval()

        inp = open("test.json", "r", encoding="utf-8")
        passages = json.load(inp)
        end = len(passages)
        for begin in range(0, end, args.step):
            if passages[min(end - 1, begin + args.step - 1)].get("label", None) == None:
                break
        start = begin
        inp.close()

        for begin in range(start, end, args.step):
            test_loader = Data.DataLoader(
                dataset=CustomDataset(path="test.json", sen_num=args.sen_num, begin=begin, end=begin + args.step),
                batch_size=args.batch_size, shuffle=False)
            pred, val = run(data_loader=test_loader, update_model=False, predict=True)
            inp = open("test.json", "r", encoding="utf-8")
            passages = json.load(inp)
            for i in range(pred.shape[0]):
                # print(pred[i].item())
                passages[begin + i]['label'] = pred[i].item()
                passages[begin + i]['semantic_value'] = val[i].detach().numpy().tolist()
            inp.close()

            outp = open("test.json", 'w', encoding="utf-8")
            outp.write(json.dumps(passages, indent=4, ensure_ascii=False))
            outp.close()

            print("{}/{}".format(begin, end))
