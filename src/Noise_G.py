import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import scipy
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score # Sequence labeling evaluation tool
from transformers import AutoTokenizer
import pdb

from src.dataloader import *
from src.utils import *

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index  # -100



class Noise_G_model(object):
    def __init__(self, params, refer_model, label_list, input_dim, num_heads, dropout, num_layers):
        # parameters
        self.params = params # Configuration
        self.label_list = label_list
        self.refer_model = refer_model
        self.noise_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, input_dim, dropout),
            num_layers) # 4 layers

        # training
        self.lr = float(params.lr_noise)
        self.mu = 0.9
        self.weight_decay = 5e-4


    def batch_forward(self, all_features):
        # Compute features

        self.all_features = all_features
        self.noise = self.noise_model(self.all_features)
        self.noise_input = self.all_features + self.noise
        return self.noise
        # # Compute logits logits
        # self.logits = self.model.forward_classifier(self.all_features[1][-1])   # (bsz, seq_len, output_dim)


    # def batch_loss(self, labels):
    #     '''
    #         Cross-Entropy Loss
    #     '''
    #     self.loss = 0
    #     assert self.logits!=None, "logits is none!"

    #     # classification loss
    #     ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]),
    #                             labels.flatten().long()) # bs*seq_len, out_dim; ignores -100 labels by default (pad, cls, sep, and later subword indices)
    #     self.loss = ce_loss
    #     return ce_loss.item()



    def batch_loss_noise(self, labels, cs_factor, l2_factor, d_factor):

        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim # old model
        # all_dims = self.model.classifier.output_dim
        assert self.refer_model!=None, "refer_model is none!"
        # pdb.set_trace()

        # with torch.no_grad():
        self.refer_model.eval()
        refer_all_features= self.all_features
        # refer_features = refer_all_features[1][-1]
        fea_ref = self.refer_model.encoder.bert.encoder(refer_all_features, output_attentions=True,output_hidden_states=True)
        # features = self.encoder(X)
        refer_features = fea_ref[1][-1]
        refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)

        noise_all_features= self.noise_input
        # attack_Emb = refer_features + noise
        # noise_features = noise_all_features
        fea_noise = self.refer_model.encoder.bert.encoder(noise_all_features, output_attentions=True,output_hidden_states=True)
        # features = self.encoder(X)
        noise_features = fea_noise[1][-1]
        # noise_features = noise_all_features[1][-1]
        noise_logits = self.refer_model.forward_classifier(noise_features)# (bsz,seq_len,refer_dims)


        # Check input
        assert refer_logits!=None, "refer_logits is none!"
        assert noise_logits!=None, "noise_logits is none!"

        # L2 norm loss
        loss_l2 = nn.MSELoss(reduction='mean')(self.noise_input,self.all_features)
        loss_l2 = loss_l2 * l2_factor
        # loss_l2 = torch.norm(self.noise, p=2, dim=-1).mean() # dim=0, loss.shape = 1. dim=1, loss.shape = bsz * dims. dim=-1, shape = bsz * token_len

        # cs loss

        refer_label = torch.ones_like(labels).to(device=labels.device)
        refer_label = refer_label * 1
        label_cs = torch.where(labels > 0, refer_label, labels).to(device=labels.device)


        probs = torch.softmax(refer_logits, dim=-1) # (bsz,seq_len,class_num)
        _, predict_labels = probs.max(dim=-1) # max prob and their index

        probs_e = torch.softmax(noise_logits, dim=-1) # (bsz,seq_len,class_num)
        _, noise_predict_labels = probs_e.max(dim=-1) # max prob and their index
        # pdb.set_trace()

        loss_cs = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')(probs_e.permute(0,2,1), label_cs)
        loss_cs = cs_factor * loss_cs

        # d loss
        # mask_o = (noise_predict_labels != 0) & (labels != 0) & (labels != -100)
        mask_o = (labels != 0) & (labels != -100)
        mask_d = ~mask_o

        p_loss = F.kl_div(probs.log(), probs_e, reduction='none',log_target=True)
        q_loss = F.kl_div(probs_e.log(), probs, reduction='none',log_target=True)

        # pdb.set_trace()

        p_loss.masked_fill_(mask_d.unsqueeze(2), 0.)
        q_loss.masked_fill_(mask_d.unsqueeze(2), 0.)

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        # loss_kl = (p_loss + q_loss) / 2
        # loss_kl = loss_kl / 8
        loss_kl = q_loss

        mask_sum = (mask_o*1).sum()
        # pdb.set_trace()
        if mask_sum != 0:
            loss_kl = loss_kl / mask_sum

        if loss_kl < 1e-4 and loss_kl != 0.0:
            loss_d = torch.tensor([10000.0]).to(device=loss_kl.device)
        elif loss_kl == 0.0:
            loss_d = torch.tensor([0.0]).to(device=loss_kl.device)
        else:
            loss_d = 1.0 / loss_kl

        loss_d = loss_d * d_factor


        self.loss = loss_l2 + loss_cs + loss_d
        # pdb.set_trace()

        return loss_l2.item(), loss_cs.item(), loss_d.item()



    def batch_backward(self):
        self.noise_model.train()
        self.optimizer.zero_grad()
        # for name, parms in self.noise_model.named_parameters():
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print("===")

        self.loss.backward()
        # pdb.set_trace()
        self.optimizer.step()
        # print("=============After Update===========")
        # for name, parms in self.noise_model.named_parameters():
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad.sum())
        #     print("===")
        # for name, parms in self.refer_model.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad.sum())
        #     print("===")
        # print(self.optimizer)
        # input("=====end=====")
        # pdb.set_trace()

        return self.loss.item()

    def evaluate(self, dataloader, each_class=False, entity_order=[], is_plot_hist=False, is_plot_cm=False):
        with torch.no_grad():
            self.noise_model.eval()
            self.refer_model.eval()

            y_list = []
            x_list = []
            logits_list = []

            for x, y in dataloader:
                # pdb.set_trace()
                x, y = x.cuda(), y.cuda()

                refer_all_features = self.refer_model.encoder.bert.embeddings(x)
                noise_fea = refer_all_features + self.noise_model(refer_all_features)
        
                fea_ref = self.refer_model.encoder.bert.encoder(noise_fea, output_attentions=True,output_hidden_states=True)
                refer_features = fea_ref[1][-1]
                logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)
        
                # fea = self.refer_model.encoder(x)
                # noise_fea = fea + self.noise_model(fea)
                # # pdb.set_trace()
                # logits = self.refer_model.forward_classifier(noise_fea)
                _logits = logits.view(-1, logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
                x_list.append(x)
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)


            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)
            pred_list = torch.argmax(logits_list, dim=-1)


            ### Plot the (logits) prob distribution for each class
            if is_plot_hist: # False
                plot_prob_hist_each_class(deepcopy(y_list),
                                        deepcopy(logits_list),
                                        ignore_label_lst=[
                                            self.label_list.index('O'),
                                            pad_token_label_id
                                        ])


            ### for confusion matrix visualization
            if is_plot_cm: # False
                plot_confusion_matrix(deepcopy(pred_list),
                                deepcopy(y_list),
                                label_list=self.label_list,
                                pad_token_label_id=pad_token_label_id)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            for pred_index, gold_index in zip(pred_list, y_list):
                gold_index = int(gold_index)
                if gold_index != pad_token_label_id: # !=-100
                    pred_token = self.label_list[pred_index] #
                    gold_token = self.label_list[gold_index]
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    pred_line.append(pred_token)
                    gold_line.append(gold_token)

            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            self.noise_model.train()

            # compute overall f1 score
            # micro f1 (default)
            f1 = f1_score([gold_line], [pred_line])*100
            # macro f1 (average of each class f1)
            ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
            if not each_class:
                return f1, ma_f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, ma_f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ma_f1, ordered_f1_score_dict

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))

        torch.save(self.noise_model.state_dict(), saved_path)
        # torch.save({
        #     "hidden_dim": self.noise_model.hidden_dim,
        #     "output_dim": self.noise_model.output_dim,
        #     "encoder": self.noise_model.state_dict(),
        # }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)
        # pdb.set_trace()
        self.noise_model.load_state_dict(ckpt)
        # pdb.set_trace()
        # self.noise_model=torch.load(load_path)


        # self.noise_model.hidden_dim = ckpt['hidden_dim']
        # self.noise_model.output_dim = ckpt['output_dim']
        # self.noise_model.load_state_dict(ckpt['encoder'])
        logger.info("Model has been load from %s" % load_path)
