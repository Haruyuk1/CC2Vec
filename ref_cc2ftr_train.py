from cProfile import label
import torch
import os, datetime
import torch 
import torch.nn as nn
from tqdm import tqdm
from ref_cc2ftr_model import HierachicalRNN
from torch.utils.data import DataLoader
from ref_utils import save

def train_model(dataset, params):
    dataloader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    params.class_num = 1
    params.word_batch_size = params.batch_size * params.code_file * params.code_hunk * params.code_line
    params.line_batch_size = params.batch_size * params.code_file * params.code_hunk
    params.hunk_batch_size = params.batch_size * params.code_file

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.BCELoss()

    # for logging
    train_size = len(dataset)
    def metrics(predict, labels):
        if predict.shape != labels.shape:
            print('shape error in func:metrics')
            return
        def count(condition):
            c = torch.where(condition,
                            torch.ones(predict.shape).to(params.device),
                            torch.zeros(predict.shape).to(params.device))
            return int(torch.sum(c))
        TP = count(torch.logical_and(predict >= 0.5, labels >= 0.5))
        FP = count(torch.logical_and(predict >= 0.5, labels < 0.5))
        TN = count(torch.logical_and(predict < 0.5, labels < 0.5))
        FN = count(torch.logical_and(predict < 0.5, labels >= 0.5))
        return (TP, FP, TN, FN)


    # batches = batches[:10] # 謎のスライス jitの方ではあるけどなんであるんだ…？
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        for labels, removed_code, added_code in tqdm(dataloader):
            # debug
            if params.cuda:
                labels = labels.cuda()
                removed_code = removed_code.cuda()
                added_code = added_code.cuda()

            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            labels = torch.cuda.FloatTensor(labels)
            optimizer.zero_grad()
            predict = model.forward(removed_code, added_code, state_hunk, state_sent, state_word)
            tmp_TP, tmp_FP, tmp_TN, tmp_FN =  metrics(predict, labels)
            TP, FP, TN, FN = TP+tmp_TP, FP+tmp_FP, TN+tmp_TN, FN+tmp_FN
            loss = criterion(predict, labels)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))                
        print(f'\tmetrics TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')
        print(f'\taccuracy:{(TP+TN)/(TP+FP+TN+FN)}')
        save(model, params.save_dir, 'epoch', epoch)