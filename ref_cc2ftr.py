import argparse
import pickle
import numpy as np
from ref_cc2ftr_eval import eval_model
from ref_utils import CodeDataset, codeDatasetWithoutLabel
from ref_cc2ftr_train import train_model

def read_args():
    parser = argparse.ArgumentParser()

    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    parser.add_argument('-train_data', type=str, default='./data/ref/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/ref/test.pkl', help='the directory of our testing data')
    parser.add_argument('-dictionary_data', type=str, default='./data/ref/dict.pkl', help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-eval', action='store_true', help='extracting features')
    parser.add_argument('-eval_data', type=str, help='the directory of our extracting data')
    parser.add_argument('-name', type=str, help='name of our output file')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('--code_file', type=int, default=10, help='the number of files in commit code')
    parser.add_argument('--code_hunk', type=int, default=5, help='the number of hunks in each file')
    parser.add_argument('--code_line', type=int, default=20, help='the number of LOC in each hunk of commit code')
    parser.add_argument('--code_length', type=int, default=40, help='the length of each LOC of commit code')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for Attention model
    parser.add_argument('-valid_ratio', type=float, default=0.1, help='the ratio of valid data')
    parser.add_argument('-embed_size', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=32, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=0, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')    

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()    
    
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        train_labels, train_codes = train_data    

        labels = np.array([1 if train_label else 0 for train_label in train_labels])
        codes = train_codes
        
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_code = dictionary

        max_settings = (params.code_file, params.code_hunk, params.code_line, params.code_length)
        dataset = CodeDataset(codes, labels, dict_code, max_settings)
        params.vocab_code = len(dict_code)

        train_model(dataset=dataset, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
    
    elif params.eval is True:
        eval_data = pickle.load(open(params.eval_data, 'rb'))
        eval_labels, eval_codes = eval_data

        if eval_labels:
            labels = np.array([1 if eval_label else 0 for eval_label in eval_labels])
        else:
            labels = None
        codes = eval_codes

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_code = dictionary

        max_settings = (params.code_file, params.code_hunk, params.code_line, params.code_length)
        if labels is not None:
            dataset = CodeDataset(codes, labels, dict_code, max_settings)
        else:
            dataset = codeDatasetWithoutLabel(codes, dict_code, max_settings)

        params.batch_size = 1
        params.vocab_code = len(dict_code)

        eval_model(dataset=dataset, params=params)


        # TODO 
        # pad_msg = padding_message(data=msgs, max_length=params.msg_length)
        # added_code, removed_code = clean_and_reformat_code(codes)
        # pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
        # pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

        # pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
        # pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
        # pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
        # pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
        
        # data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)
        # params.batch_size = 1
        # extracted_cc2ftr(data=data, params=params)
        # print('--------------------------------------------------------------------------------')
        # print('--------------------------Finish the extracting process-------------------------')
        # print('--------------------------------------------------------------------------------')
        # exit()
