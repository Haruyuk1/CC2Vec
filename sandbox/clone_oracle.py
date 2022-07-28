import json
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import git
from tqdm import tqdm


def cloneFromUrl(repo_url, save_dir):
    name = repo_url.split('/')[-1].replace('.git', '')
    to_path = os.path.join(save_dir, name)
    if os.path.exists(to_path):
        print(f'{name} already exists, so skipped.')
        return None

    try:
        repo = git.Repo.clone_from(url=repo_url, to_path=to_path)
        print(f'{name} is successfly cloned.')
    except Exception as e:
        print(e)
        print(f'repository {repo_url} was skipped because of an exeption.')
        return None
    return repo

    
def main():
    parser = ArgumentParser()
    parser.add_argument('-max_workers', type=int, default=12)    
    parser.add_argument('-repos', type=str, default='repos')

    params = parser.parse_args()

    with open('data/sandbox/oracle.json', 'rb') as f_oracle:
        oracle_set = json.load(f_oracle)
    
    repo_url_list = set()
    for data in oracle_set:
        repo_url_list.add(data['repository'])
    iterable_save_dir = [params.repos] * len(repo_url_list)

    with ProcessPoolExecutor(params.max_workers) as executor:
        repos = list(tqdm(executor.map(cloneFromUrl, repo_url_list, iterable_save_dir), total=len(repo_url_list)))
        
    return


if __name__ == "__main__":
    main()
