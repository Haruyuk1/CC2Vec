import json
from multiprocessing.pool import ThreadPool
import pprint
import subprocess
import re

import pandas as pd
from tqdm import tqdm


RMCMD = 'RefactoringMiner'
NUM_POOL = 20


def refMine(args):
    repo_name = args[0]
    commit_sha1 = args[1]
    cp = subprocess.run([RMCMD, '-c', repo_name, commit_sha1], encoding='utf-8', capture_output=True)
    if cp.returncode != 0:
        print(f'[fialed] repo_name:{repo_name}, commit_sha1:{commit_sha1}')
        return

    match = re.search(
        r'''
        repo:
        (.+?)
        ,\s
        commit:
        (.+?)
        ,\s
        refactorings:
        ([0-9]*?)
        \n
        ''', cp.stdout, flags=re.VERBOSE+re.MULTILINE+re.DOTALL)
    
    if not match:
        print('[failed] not match')
        return

    repo_name = match.groups()[0]
    commit_sh1 = match.groups()[1]
    refactorings = match.groups()[2]

    return int(refactorings) > 0


def main():
    data_path = 'data/sandbox/all_commits.json'
    with open(data_path, mode='r', encoding='utf-8') as f_data:
        all_commits = json.load(f_data)
    all_commits_dict = dict()
    for commit in all_commits:
        all_commits_dict[commit['sha1']] = commit
    length = len(all_commits)

    args = [(commit['repository'], commit['sha1']) for commit in all_commits_dict.values()]

    # for commit in tqdm(all_commits):
        # args = (commit['repository'], commit['sha1'])
        # refMine(args)

    with ThreadPool(NUM_POOL) as pool:
        imap = pool.imap(refMine, args) 
        result = list(tqdm(imap, total=length))
    
    return


if __name__ == "__main__":
    main()