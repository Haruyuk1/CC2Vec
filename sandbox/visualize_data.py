import pickle
import sys

from matplotlib import pyplot as plt


def main(file_path):
    with open(file_path, mode='rb') as f_train:
        label, codes = pickle.load(f_train)
    
    word_dist = dict()
    line_dist = dict()
    hunk_dist = dict()
    file_dist = dict()
    for file_changes in codes:
        num_files = len(file_changes)
        if num_files in file_dist:
            file_dist[num_files] += 1
        else:
            file_dist[num_files] = 1
        for hunk_changes in file_changes:
            num_hunks = len(hunk_changes)
            if num_hunks in hunk_dist:
                hunk_dist[num_hunks] += 1
            else:
                hunk_dist[num_hunks] = 1
            for hunk_change in hunk_changes:
                num_removed_lines = len(hunk_change['removed_code'])
                num_added_lines = len(hunk_change['added_code'])
                if num_removed_lines in line_dist:
                    line_dist[num_removed_lines] += 1
                else:
                    line_dist[num_removed_lines] = 1
                if num_added_lines in line_dist:
                    line_dist[num_added_lines] += 1
                else:
                    line_dist[num_added_lines] = 1
                for line in hunk_change['removed_code']:
                    num_words = len(line.split(' '))
                    if num_words in word_dist:
                        word_dist[num_words] += 1
                    else:
                        word_dist[num_words] = 1
                for line in hunk_change['added_code']:
                    num_words = len(line.split(' '))
                    if num_words in word_dist:
                        word_dist[num_words] += 1
                    else:
                        word_dist[num_words] = 1
    
    def dictToXY(dic: dict):
        X, Y = [], []
        for x, y in dic.items():
            X.append(x)
            Y.append(y)
        return X, Y

    word_x, word_y = dictToXY(word_dist)
    line_x, line_y = dictToXY(line_dist)
    hunk_x, hunk_y = dictToXY(hunk_dist)
    file_x, file_y = dictToXY(file_dist)

    figure = plt.figure()
    ax1 = figure.add_subplot(2,2,1)
    ax2 = figure.add_subplot(2,2,2)
    ax3 = figure.add_subplot(2,2,3)
    ax4 = figure.add_subplot(2,2,4)

    ax1.bar(word_x, word_y)
    ax2.bar(line_x, line_y)
    ax3.bar(hunk_x, hunk_y)
    ax4.bar(file_x, file_y)

    ax1.set_xlim([0, 50])
    ax2.set_xlim([0, 50])
    ax3.set_xlim([0, 25])
    ax4.set_xlim([0, 50])

    ax1.set_title('word distribution in line')
    ax2.set_title('line distribution in hunk')
    ax3.set_title('hunk distribution in file')
    ax4.set_title('file distribution in commit')

    plt.show()
 

    return


if __name__ == "__main__":
    main(sys.argv[1])
