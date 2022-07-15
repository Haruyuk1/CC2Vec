import argparse
import pickle
import json
from pydoc import describe
import re
from pprint import pprint

import git
from git.compat import defenc
from pkg_resources import parse_requirements
from tqdm import tqdm


RE_COMMENT_PATTERN = re.compile(r'/\*(.*?)\*/', flags=re.DOTALL+re.MULTILINE)
RE_HUNK_PATTERN = re.compile(r'@@.*?@@(.*?)[(@@)|$]', flags=re.DOTALL+re.MULTILINE+re.VERBOSE)
RE_TORKNIZE_PATTERN = re.compile(r'([\{\}\(\)\[\]\.,";])')


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


def torknizeJavaLine(line: str) -> str:
    tokens = re.split(RE_TORKNIZE_PATTERN, line)
    concat_tokens = " ".join(tokens)
    return re.sub(r'\s{1,}', ' ', concat_tokens)


def excludeComment(text: str) -> str:
    return re.sub(RE_COMMENT_PATTERN, '', text)
    

def makeJavaChangeSet(commit: git.Commit):
    file_changes: list[list[dict]] = [] # for each file
    diff_index: git.DiffIndex = commit.diff(commit.parents[0], create_patch=True)
    for diff in diff_index:
        hunk_changes: list[dict] = [] # for each hunk
        hunk_change: dict = dict()
        hunk_change['removed_code'] = []
        hunk_change['added_code'] = []
        diff: git.Diff = diff
        if not isJavaChange(diff):
            continue

        diff_text = diff.diff.decode(defenc)
        # diff_text = excludeComment(diff_text) # 多分構文解析しないとコメントは弾けない
        diff_lines = diff_text.split('\n')
        for diff_line in diff_lines:
            if diff_line.startswith('-'):
                hunk_change['removed_code'].append(torknizeJavaLine(diff_line[1:]))
            if diff_line.startswith('+'):
                hunk_change['added_code'].append(torknizeJavaLine(diff_line[1:]))
            if diff_line.startswith('@@'): # hunk separation
                if hunk_change['removed_code'] or hunk_change['added_code']:
                    hunk_changes.append(hunk_change)
                    hunk_change = {'removed_code': [], 'added_code': []}
        if hunk_change['removed_code'] or hunk_change['added_code']:
            hunk_changes.append(hunk_change)
        file_changes.append(hunk_changes)
    
    return file_changes if file_changes else None


def main():
    params = read_args().parse_args()

    with open(params.data, mode='r', encoding='utf-8') as f_data:
        labeled_commits = json.load(f_data)

    # repositoryのsetを抽出
    local_repos = set()
    for commit in labeled_commits['commits']:
        github_repo = commit['repository']
        local_repo = params.repos + '/' + re.search(r'//(.+)/(.+?).git', github_repo).groups()[-1]
        if not local_repo in local_repos:
            local_repos.add(local_repo)

    # Repositoryオブジェクトの辞書
    repo_dict = dict()
    for local_repo in local_repos:
        repo_dict[local_repo] = git.Repo(local_repo)

    # labelsの構築
    labels = [len(commit['refactorings']) > 0 for commit in labeled_commits['commits']]

    # codesの構築
    codes = []
    for commit in tqdm(labeled_commits['commits'], desc='codes construct'):
        github_repo = commit['repository']
        local_repo = params.repos + '/' + re.search(r'//(.+)/(.+?).git', github_repo).groups()[-1]
        repo: git.Repo = repo_dict[local_repo]
        git_commit: git.Commit = repo.commit(commit['sha1'])
        data = makeJavaChangeSet(git_commit)
        if data:
            codes.append(data)

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
