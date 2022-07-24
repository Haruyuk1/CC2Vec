import argparse
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import git
import javalang
from javalang import tokenizer
from tqdm import tqdm

RE_COMMENT_PATTERN = re.compile(r'/\*(.*?)\*/', flags=re.DOTALL+re.MULTILINE)
RE_HUNK_PATTERN = re.compile(r'''
    @@\s-([0-9]*?),([0-9]*?)\s\+([0-9]*?),([0-9]*?)\s@@.*?\n # 1行目
    (.*?) # 2行目以降
    (?=@@|$)
    ''', flags=re.DOTALL+re.VERBOSE)
RE_TORKNIZE_PATTERN = re.compile(
    r'''[
    \{\}\(\)\[\]<>
    \+\-\*\/
    =\.,";
    ]''', flags=re.MULTILINE+re.VERBOSE)

NUM_SKIPPED = 0

def read_args():
    parser = argparse.ArgumentParser()

    # labeled_all_commits
    parser.add_argument('-data')
    
    # repository directory
    parser.add_argument('-repos', default='repos')

    # pickle save directory and name
    parser.add_argument('-save', default='data/ref')
    parser.add_argument('-name_data', default='train.pkl')
    parser.add_argument('-name_dict', default='dict.pkl')
    
    # for debug
    parser.add_argument('-debug', action='store_true', default=False)

    return parser


def isJavaChange(diff: git.Diff):
        res = True
        if diff.a_blob:
            if not diff.a_blob.name.endswith('.java'):
                res = False
        if diff.b_blob:
            if not diff.b_blob.name.endswith('.java'):
                res = False
        return res


def replaceToken(tokens: 'list[tokenizer.JavaToken]'):
    type_should_be_replaced = \
        (tokenizer.Literal, tokenizer.Integer, tokenizer.FloatingPoint)
    def replaceIfLiteral(token: tokenizer.JavaToken):
        if isinstance(token, type_should_be_replaced):
            token.value = '<LITERAL>'
        return token
    return list(map(replaceIfLiteral, tokens))


def splitToken(tokens: 'list[tokenizer.JavaToken]'):
    def splitIfIdentifier(token: tokenizer.JavaToken):
        if not isinstance(token, tokenizer.Identifier):
            return token
        words = token.value
        if '_' in words: # snake_case
            token.value = ' '.join(words.split('_'))
            return token
        splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', token.value)).split()
        token.value = ' '.join(splitted)
        return token
    return list(map(splitIfIdentifier, tokens))

def makeJavaChangeSet(commit, local_repo):
    repo: git.Repo = git.Repo(local_repo)
    git_commit: git.Commit = repo.commit(commit['sha1'])
    file_changes: list[list[dict]] = [] # for each file
    diff_index: git.DiffIndex = git_commit.diff(git_commit.parents[0], create_patch=True)
    
    # ファイルごとの処理
    for diff in diff_index:
        hunk_changes: list[dict] = [] # for each hunk
        diff: git.Diff = diff
        if not isJavaChange(diff):
            continue

        try:
            unified_diff = diff.diff.decode('utf-8')
        except Exception as e:
            global NUM_SKIPPED
            NUM_SKIPPED += 1
            return None

        a_blob_tokens = [] # a_blobの全トークン
        b_blob_tokens = [] # b_blobの全トークン

        try:
            if diff.b_blob:
                code = diff.b_blob.data_stream.read().decode('utf-8')
                b_blob_tokens = list(javalang.tokenizer.tokenize(code))
            if diff.a_blob:
                code = diff.a_blob.data_stream.read().decode('utf-8')
                a_blob_tokens = list(javalang.tokenizer.tokenize(code))
        except Exception as e:
            NUM_SKIPPED += 1
            return None

        # トークンごとの処理
        a_blob_tokens = replaceToken(a_blob_tokens)
        b_blob_tokens = replaceToken(b_blob_tokens)
        a_blob_tokens = splitToken(a_blob_tokens)
        b_blob_tokens = splitToken(b_blob_tokens)

        # hunkごとに処理    
        matches = re.findall(RE_HUNK_PATTERN, unified_diff)

        for match in matches:
            a_index, a_range, b_index, b_range, diff_lines = match
            a_index, b_index = int(a_index), int(b_index)
            removed_line_indexes = [] # 削除された行
            added_line_indexes = [] # 追加された行
            hunk_change: dict = dict()
            hunk_change['removed_code'] = []
            hunk_change['added_code'] = []

            # 変更行の抽出
            for line in diff_lines.split('\n'):
                line: str = line
                if line.startswith('-'):
                    removed_line_indexes.append(a_index)
                    a_index += 1
                elif line.startswith('+'):
                    added_line_indexes.append(b_index)
                    b_index += 1
                else:
                    a_index += 1
                    b_index += 1

            for index in removed_line_indexes:
                removed_tokens = [token.value for token in a_blob_tokens if token.position[0] == index]
                if removed_tokens:
                    hunk_change['removed_code'].append(' '.join(removed_tokens))
            for index in added_line_indexes:
                added_tokens = [token.value for token in b_blob_tokens if token.position[0] == index]
                if added_tokens:
                    hunk_change['added_code'].append(' '.join(added_tokens))
            
            if hunk_change['removed_code'] or hunk_change['added_code']:
                hunk_changes.append(hunk_change)

        if hunk_changes:   
            file_changes.append(hunk_changes)
    
    return file_changes if file_changes else None


def main():
    params = read_args().parse_args()

    with open(params.data, mode='r', encoding='utf-8') as f_data:
        labeled_commits = json.load(f_data)

    # labelsの構築
    labels = [len(commit['refactorings']) > 0 for commit in labeled_commits['commits']]

    # codesの構築
    codes = []
    local_repos = []
    for commit in labeled_commits['commits']:
        github_repo = commit['repository']
        local_repo = params.repos + '/' + re.search(r'//(.+)/(.+?).git', github_repo).groups()[-1]
        local_repos.append(local_repo)
    # 並列化
    with tqdm(total=len(labeled_commits['commits'])) as pbar:
        with ProcessPoolExecutor(6) as executor:
            codes = list(tqdm(executor.map(
                makeJavaChangeSet, labeled_commits['commits'], local_repos
                ), desc='code construct', total=len(labeled_commits['commits'])))

    if NUM_SKIPPED:
        print(f'{NUM_SKIPPED} commits were skipped because of exception.')

    # 例外が発生したコミットを除去
    tmp_labels, tmp_codes = [], []
    for label, code in zip(labels, codes):
        if code == None:
            continue
        tmp_labels.append(label)
        tmp_codes.append(code)
    labels = tmp_labels
    codes = tmp_codes

    assert len(labels) == len(codes)

    # dictionaryの構築
    bag_of_words = dict()
    dictionary: dict = dict() # str -> int(id)
    for file_changes in tqdm(codes, desc='dictionary construct'):
        for hunk_changes in file_changes:
            for hunk_change in hunk_changes:
                for line in hunk_change['removed_code']:
                    for word in line.split(' '):
                        if word in bag_of_words:
                            bag_of_words[word] += 1
                        else:
                            bag_of_words[word] = 1
                for line in hunk_change['added_code']:
                    for word in line.split(' '):
                        if word in bag_of_words:
                            bag_of_words[word] += 1
                        else:
                            bag_of_words[word] = 1
    for index, word in enumerate(bag_of_words, start=1):
        dictionary[word] = index

    dataset = (labels, codes)
    
    if params.debug:
        with open('data/sandbox/trial_dataset.json', mode='w', encoding='utf-8') as f_json:
            json.dump(codes, f_json, indent='\t')
        return

    with open(params.save + '/' + params.name_data, mode='wb') as f_data, \
         open(params.save + '/' + params.name_dict, mode='wb') as f_dict:
        pickle.dump(dataset, f_data)
        pickle.dump(dictionary, f_dict)

    return


if __name__ == "__main__":
    main()
