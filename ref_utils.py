import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, codes, labels, dictionary, max_settings):
        self.codes = codes
        self.labels = torch.Tensor(labels).cuda()
        self.dictionary = dictionary
        self.max_settings = max_settings
        self.dictionary['<NULL>'] = 0

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index: int):
        tmp = self.fileChanges2Tensor(self.codes[index])
        removed_code = tmp[0]
        added_code = tmp[1]
        return self.labels[index], removed_code, added_code

    def fileChanges2Tensor(self, file_changes):
        max_file, max_hunk, max_line, max_length = self.max_settings
        removed_code_tensor = torch.zeros(size=(max_file, max_hunk, max_line, max_length), dtype=torch.int)
        added_code_tensor = torch.zeros(size=(max_file, max_hunk, max_line, max_length), dtype=torch.int)
        for f, hunk_changes in enumerate(file_changes[:min(len(file_changes), max_file)]):
            for h, hunk_change in enumerate(hunk_changes[:min(len(hunk_changes), max_hunk)]):
                removed_lines = hunk_change['removed_code']
                added_lines = hunk_change['added_code']
                for l, line in enumerate(removed_lines[:min(len(removed_lines), max_line)]):
                    words = line.split()
                    for w, word in enumerate(words[:min(len(words), max_length)]):
                        removed_code_tensor[f, h, l, w] = self.str2Int(word)
                for l, line in enumerate(added_lines[:min(len(added_lines), max_line)]):
                    words = line.split()
                    for w, word in enumerate(words[:min(len(words), max_length)]):
                        added_code_tensor[f, h, l, w] = self.str2Int(word)
        return removed_code_tensor, added_code_tensor

    def str2Int(self, s: str) -> int:
        if s in self.dictionary:
            return self.dictionary[s]
        else:
            return self.dictionary['<NULL>']


class codeDatasetWithoutLabel(Dataset): # FIXME duplicated code
    def __init__(self, codes, dictionary, max_settings):
        self.codes = codes
        self.dictionary = dictionary
        self.max_settings = max_settings
        self.dictionary['<NULL>'] = 0

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index: int):
        tmp = self.fileChanges2Tensor(self.codes[index])
        removed_code = tmp[0]
        added_code = tmp[1]
        return removed_code, added_code
        
    def fileChanges2Tensor(self, file_changes):
        max_file, max_hunk, max_line, max_length = self.max_settings
        removed_code_tensor = torch.zeros(size=(max_file, max_hunk, max_line, max_length), dtype=torch.int)
        added_code_tensor = torch.zeros(size=(max_file, max_hunk, max_line, max_length), dtype=torch.int)
        for f, hunk_changes in enumerate(file_changes[:min(len(file_changes), max_file)]):
            for h, hunk_change in enumerate(hunk_changes[:min(len(hunk_changes), max_hunk)]):
                removed_lines = hunk_change['removed_code']
                added_lines = hunk_change['added_code']
                for l, line in enumerate(removed_lines[:min(len(removed_lines), max_line)]):
                    words = line.split()
                    for w, word in enumerate(words[:min(len(words), max_length)]):
                        removed_code_tensor[f, h, l, w] = self.str2Int(word)
                for l, line in enumerate(added_lines[:min(len(added_lines), max_line)]):
                    words = line.split()
                    for w, word in enumerate(words[:min(len(words), max_length)]):
                        added_code_tensor[f, h, l, w] = self.str2Int(word)
        return removed_code_tensor, added_code_tensor

    def str2Int(self, s: str) -> int:
        if s in self.dictionary:
            return self.dictionary[s]
        else:
            return self.dictionary['<NULL>']


def metrics(predict, labels, device):
        if predict.shape != labels.shape:
            print('shape error in func:metrics')
            return
        def count(condition):
            c = torch.where(condition,
                            torch.ones(predict.shape).to(device),
                            torch.zeros(predict.shape).to(device))
            return int(torch.sum(c))
        TP = count(torch.logical_and(predict >= 0.5, labels >= 0.5))
        FP = count(torch.logical_and(predict >= 0.5, labels < 0.5))
        TN = count(torch.logical_and(predict < 0.5, labels < 0.5))
        FN = count(torch.logical_and(predict < 0.5, labels >= 0.5))
        return (TP, FP, TN, FN)


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)
