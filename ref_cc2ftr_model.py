import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


# The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(WordRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # Word Encoder
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.wordRNN = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        # Word Attention
        self.wordattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 1, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)
        out_state, hid_state = self.wordRNN(emb_out, hid_state)
        out_state_t = out_state.transpose(1,2)
        word_annotation = self.wordattn(out_state)
        word_annotation = F.relu(word_annotation)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)
        sent = torch.bmm(out_state_t, attn).squeeze(2)
        return sent, hid_state


# The sentence RNN model for generating a hunk vector
class SentRNN(nn.Module):
    def __init__(self, sent_size, hidden_size):
        super(SentRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.sentRNN = nn.GRU(sent_size, hidden_size, bidirectional=True, batch_first=True)

        # Sentence Attention
        self.sentattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 1, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.sentRNN(inp, hid_state)
        out_state_t = out_state.transpose(1,2)
        sent_annotation = self.sentattn(out_state)
        sent_annotation = F.relu(sent_annotation)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)
        hunk = torch.bmm(out_state_t, attn).squeeze(2)
        return hunk, hid_state


# The hunk RNN model for generating the vector representation for the instance
class HunkRNN(nn.Module):
    def __init__(self, hunk_size, hidden_size):
        super(HunkRNN, self).__init__()
        # Sentence Encoder
        self.hunk_size = hunk_size
        self.hunkRNN = nn.GRU(hunk_size, hidden_size, bidirectional=True, batch_first=True)

        # Sentence Attention
        self.hunkattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 1, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.hunkRNN(inp, hid_state)
        out_state_t = out_state.transpose(1,2)
        hunk_annotation = self.hunkattn(out_state)
        hunk_annotation = F.relu(hunk_annotation)
        attn = F.softmax(self.attn_combine(hunk_annotation), dim=1)
        hunks = torch.bmm(out_state_t, attn).squeeze(2)
        return hunks, hid_state


# The HAN model
class HierachicalRNN(nn.Module):
    def __init__(self, args):
        super(HierachicalRNN, self).__init__()
        self.vocab_size = args.vocab_code
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.code_file = args.code_file
        self.word_batch_size = args.word_batch_size
        self.line_batch_size = args.line_batch_size
        self.hunk_batch_size = args.hunk_batch_size
        self.cls = args.class_num

        self.dropout = nn.Dropout(args.dropout_keep_prob)  # drop out

        # Word Encoder
        self.wordRNN = WordRNN(self.vocab_size, self.embed_size, self.hidden_size)
        # Sentence Encoder
        self.sentRNN = SentRNN(2 * self.hidden_size, self.hidden_size)
        # Hunk Encoder
        self.hunkRNN = HunkRNN(2 * self.hidden_size, self.hidden_size)

        # standard neural network layer
        self.standard_nn_layer = nn.Linear(4 * self.hidden_size, self.hidden_size)

        # neural network tensor
        self.W_nn_tensor_one = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.W_nn_tensor_two = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.V_nn_tensor = nn.Linear(4 * self.hidden_size, 2)

        # Hidden layers before putting to the output layer
        self.fc1 = nn.Linear(self.code_file * (5 * self.hidden_size + 4), 2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.cls)
        self.sigmoid = nn.Sigmoid()

    def forward_code(self, x: torch.Tensor, hid_state):
        hid_state_hunk, hid_state_sent, hid_state_word = hid_state
        n_batch, n_file, n_hunk, n_line, n_word = x.shape
        
        x = x.reshape(n_batch*n_file*n_hunk*n_line, n_word).cuda()
        sent, state_word = self.wordRNN(x, hid_state_word)
        sent = sent.reshape(n_batch*n_file*n_hunk, n_line, 2 * self.hidden_size).cuda()
        hunk, state_sent = self.sentRNN(sent, hid_state_sent)
        hunk = hunk.reshape(n_batch*n_file, n_hunk, 2 * self.hidden_size).cuda()
        hunks, state_hunk = self.hunkRNN(hunk, hid_state_hunk)

        change_repr = hunks.reshape(n_batch, n_file, 2 * self.hidden_size)

        return change_repr # [batch_size, code_file, 2*hidden_size]


    def forward(self, removed_code, added_code, hid_state_hunk, hid_state_sent, hid_state_word):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)

        # [batch_size, code_file, 2*hidden_size]
        x_added_code = self.forward_code(x=added_code, hid_state=hid_state)
        x_removed_code = self.forward_code(x=removed_code, hid_state=hid_state)

        # [batch_size, code_file, 2*hidden_size]
        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        # [batch_size, code_file, 2*hidden_size]
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        # [batch_size, code_file, 1]
        cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        # [batch_size, code_file, 1]
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        # [batch_size, code_file, hidden_size]
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        # [batch_size, code_file, 2]
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        # [batch_size, code_file, 5*hidden_size+4]
        x_diff_code = torch.cat((subtract, multiple, cos, euc, nn, ntn), dim=2)
        # [batch_size, code_file*(5*hidden_size+4)]
        x_diff_code = x_diff_code.reshape(shape=(self.batch_size, self.code_file*(5*self.hidden_size+4)))
        x_diff_code = self.dropout(x_diff_code)
        
        out = self.fc1(x_diff_code)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(dim=2, eps=1e-6)
        return cosine(added_code, removed_code).view(self.batch_size, self.code_file, 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(self.batch_size, self.code_file, 1)

    def standard_neural_network_layer(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=2)
        concat = concat.reshape(shape=(self.batch_size * self.code_file, 4 * self.hidden_size))
        output = self.standard_nn_layer(concat)
        output = F.relu(output)
        output = output.reshape(shape=(self.batch_size, self.code_file, self.hidden_size))
        return output

    def neural_network_tensor_layer(self, added_code, removed_code):
        output_one = self.W_nn_tensor_one(removed_code)
        output_one = torch.mul(output_one, added_code)
        # [batch_size, code_file, 1]
        output_one = torch.sum(output_one, dim=2).reshape(shape=(self.batch_size, self.code_file, 1))

        output_two = self.W_nn_tensor_two(removed_code)
        output_two = torch.mul(output_two, added_code)
        # [batch_size, code_file, 1]
        output_two = torch.sum(output_two, dim=2).reshape(shape=(self.batch_size, self.code_file, 1))

        # [batch_size, code_file, 2]
        W_output = torch.cat((output_one, output_two), dim=2)
        # [batch_size, code_file, 4*hidden_size]
        code = torch.cat((removed_code, added_code), dim=2)
        # [batch_size, code_file, 2]
        V_output = self.V_nn_tensor(code)
        return F.relu(W_output + V_output) # [batch_size, code_file, 2]

    def init_hidden_hunk(self):
        return Variable(torch.zeros(2, self.hunk_batch_size, self.hidden_size)).cuda()

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.line_batch_size, self.hidden_size)).cuda()

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.word_batch_size, self.hidden_size)).cuda()
