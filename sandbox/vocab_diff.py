import pickle
import random
import sys

from torch.utils.data import random_split



def main(data_path):
    label, codes = pickle.load(open(data_path, 'rb'))
    
    random.shuffle(codes)

    valid_size = int(len(codes) * 0.1)
    train_size = len(codes) - valid_size
    valid_codes = codes[:valid_size]
    train_codes = codes[valid_size:]

    valid_words = dict_words(valid_codes)
    train_words = dict_words(train_codes)

    # fliter
    threashold = 10
    valid_words = set([k for k, v in valid_words.items() if v >= threashold])
    train_words = set([k for k, v in train_words.items() if v >= threashold])

    union_set = valid_words | train_words
    common_set = valid_words & train_words
    valid_uniqe = valid_words - train_words
    train_uniqe = train_words - valid_words

    print(f'valid word: {len(valid_words)}, train word: {len(train_words)}')
    print(f'valid uniqe word: {len(valid_uniqe)}, train uniqe word: {len(train_uniqe)},'\
          f'common word: {len(common_set)}, all word: {len(union_set)}')        

    return

def dict_words(codes):
    word_dict = dict()
    for commit in codes:    
        for hunk_changes in commit:
            for hunk_change in hunk_changes:
                removed_lines = hunk_change['removed_code']
                added_lines = hunk_change['added_code']
                for line in removed_lines:
                    words = line.split()
                    for word in words:
                        word_dict[word] = word_dict.get(word, 0) + 1
                for line in added_lines:
                    words = line.split()
                    for word in words:
                        word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict

if __name__ == "__main__":
    main(sys.argv[1])
