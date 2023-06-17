# coding = utf-8

import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import itertools
import time

class Pro_MDP_Net(nn.Module):
    def __init__(self, embedding_size, PathAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(Pro_MDP_Net, self).__init__()
        " The embedding of candidate and proxy workflows "
        self.PathAttention_factor = PathAttention_factor

        self.attention_W = nn.Parameter(torch.Tensor(
            embedding_size, self.PathAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.PathAttention_factor))
        self.projection_h = nn.Parameter(
            torch.Tensor(self.PathAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        " Proxy-based Matching Degree Prediction "
        in_dim = embedding_size * 5
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))


    def forward(self, inputs):
        " The embedding of candidate and proxy workflows "
        path_num = int((inputs.shape[1]-3)/2)
        candidate_input = inputs[:, 3:3+path_num, :] # candidate paths
        proxy_input = inputs[:, 3+path_num:3+path_num*2, :] # proxy paths

        candidate_temp = F.relu(torch.tensordot(
            candidate_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score1 = F.softmax(torch.tensordot(
            candidate_temp, self.projection_h, dims=([-1], [0])), dim=1)
        candidate_output = torch.sum(
            self.normalized_att_score1 * candidate_input, dim=1)

        proxy_temp = F.relu(torch.tensordot(
            proxy_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score2 = F.softmax(torch.tensordot(
            proxy_temp, self.projection_h, dims=([-1], [0])), dim=1)
        proxy_output = torch.sum(
            self.normalized_att_score2 * proxy_input, dim=1)

        " Proxy-based Matching Degree Prediction "
        QueryText_embed = inputs[:, 0, :]
        CandidateText_embed = inputs[:, 1, :]
        ProxyText_embed = inputs[:, 2, :]
        pred_input = torch.cat((QueryText_embed, CandidateText_embed, ProxyText_embed, candidate_output, proxy_output),
                               dim=1)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        out = torch.sigmoid(self.layer5(hidden_4_out))

        return out


with open('dict_des.pkl', 'rb') as f:
    dict_des = pickle.load(f)
with open('dict_proxy.pkl', 'rb') as f:
    dict_proxy = pickle.load(f)
with open('dict_path_embedding.pkl', 'rb') as f:
    dict_path_embedding = pickle.load(f)

MatchNet = Pro_MDP_Net(768, 4, 512, 512, 256, 256, 1)
MatchNet.load_state_dict(torch.load("trained_model.pt", map_location=torch.device('cpu')))
MatchNet.eval()

def np_concate(querytext_vec, candidatetext_vec, proxytext_vec, candidatepath_vec, proxypath_vec):

    a = np.expand_dims(querytext_vec, axis=0)
    b = np.expand_dims(candidatetext_vec, axis=0)
    c = np.expand_dims(proxytext_vec, axis=0)
    a = np.concatenate((a, b, c), axis=0)

    for path in candidatepath_vec:
        path_array = np.expand_dims(path, axis=0)
        a = np.concatenate((a, path_array), axis=0)
    for path in proxypath_vec:
        path_array = np.expand_dims(path, axis=0)
        a = np.concatenate((a, path_array), axis=0)

    return a

def work(tfidf_candidate, kk):

    sort = []
    for j in range(len(tfidf_candidate)):
        temp_sort = tfidf_candidate[j]
        text_id = tfidf_candidate[j][0][1]
        querytext_vec = dict_des[text_id]
        proxy_id = dict_proxy[text_id][0][0]
        proxytext_vec = dict_des[proxy_id]
        for t in range(len(tfidf_candidate[j])):
            model_id = tfidf_candidate[j][t][2]
            candidatetext_vec = dict_des[model_id]
            candidatepath_vec = dict_path_embedding[str(text_id) + ' ' + str(model_id)]
            proxypath_vec = dict_path_embedding[str(text_id) + ' ' + str(proxy_id)]
            x = np_concate(querytext_vec, candidatetext_vec, proxytext_vec, candidatepath_vec, proxypath_vec)
            x = x[None, ...]
            x = torch.from_numpy(x).float()
            out = MatchNet(x)
            out = out.detach().numpy().tolist()

            tup = temp_sort[t]
            new_tup = tuple(out) + tup
            temp_sort[t] = new_tup

        temp_sort = sorted(temp_sort, reverse=True)
        sort.append(temp_sort[0:kk])

    tfidf_sort = sort

    return tfidf_sort







