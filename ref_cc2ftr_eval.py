from audioop import add
from cProfile import label
import torch
import os, datetime
import torch 
import torch.nn as nn
from tqdm import tqdm
from ref_cc2ftr_model import HierachicalRNN
from torch.utils.data import DataLoader, random_split
from ref_utils import CodeDataset, codeDatasetWithoutLabel, metrics, save


def eval_model(dataset, params):
    eval_dataloader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=False, drop_last=False)

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    
    params.class_num = 1
    params.word_batch_size = params.batch_size * params.code_file * params.code_hunk * params.code_line
    params.line_batch_size = params.batch_size * params.code_file * params.code_hunk
    params.hunk_batch_size = params.batch_size * params.code_file

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.cuda()

    # 評価
    model.eval()
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        if isinstance(dataset, CodeDataset):
            for labels, removed_code, added_code in tqdm(eval_dataloader):
                if params.cuda:
                    labels = labels.cuda()
                    removed_code = removed_code.cuda()
                    added_code = added_code.cuda()

                # reset the hidden state of hierarchical attention model
                state_word = model.init_hidden_word()
                state_sent = model.init_hidden_sent()
                state_hunk = model.init_hidden_hunk()

                labels = torch.cuda.FloatTensor(labels)
                predict = model.forward(removed_code, added_code, state_hunk, state_sent, state_word)
                tmp_TP, tmp_FP, tmp_TN, tmp_FN = metrics(predict, labels, params.device)
                TP, FP, TN, FN = TP+tmp_TP, FP+tmp_FP, TN+tmp_TN, FN+tmp_FN
            print(f'eval metrics TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')
            print(f'eval accuracy: {(TP+TN)/(TP+FP+TN+FN)}')

        if isinstance(dataset, codeDatasetWithoutLabel):
            for removed_code, added_code in eval_dataloader:
                if params.cuda:
                    removed_code = removed_code.cuda()
                    added_code = added_code.cuda()
                
                # reset hidden state of hierarchical attention model
                state_word = model.init_hidden_word()
                state_sent = model.init_hidden_sent()
                state_hunk = model.init_hidden_hunk()
                predict = model.forward(removed_code, added_code, state_hunk, state_sent, state_word)

                # TODO ラベル無しデータの結果の出力