from argparse import ArgumentParser
import json
import git
import glob
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--head', type=int)
    params = parser.parse_args()


    repos_dir = "repos"
    repos = glob.glob(repos_dir + '/*')
    if params.head:
        repos = repos[:min(len(repos), params.head)]
    print(repos)

    def countJavaChange(commit: git.Commit):
        res = 0
        diff_index: git.DiffIndex = commit.diff(commit.parents[0])
        for diff in diff_index:
            diff: git.Diff = diff
            diff.a_blob
            b: str = diff.b_blob.name if diff.b_blob else None
            a: str = diff.a_blob.name if diff.a_blob else None
            if a:
                if a.endswith('.java'):
                    res += 1
            elif b:
                if b.endswith('.java'):
                    res += 1
        return res
    
    all_commits: list = []
    for repo_name in tqdm(repos, desc="repos loop"):
        repo_dict: dict = dict()
        commits: list[dict] = []
        repo = git.Repo(repo_name)

        repo_dict['repository'] = repo_name

        for commit in tqdm(repo.iter_commits(), desc="commit loop", leave=False):
            if len(commit.parents) != 1: # merge commit or initial commit
                continue
            commit: git.Commit = commit
            commit_dict: dict = dict()
            commit_dict['repository'] = repo_name
            commit_dict['sha1'] = commit.hexsha
            commit_dict['url'] = repo.remotes.origin.url.replace('.git', '') + '/commit/' + commit.hexsha
            commit_dict['message'] = commit.message.split('\n')[0]
            commit_dict['java_change'] = countJavaChange(commit)
            '''
            if 'merge' in commit_dict['message'].lower(): # merge commit
                print(f"{commit_dict['url']} is merge commit")
                continue
            '''
            if 'revert' in commit_dict['message'].lower(): # revert commit
                continue
            
            # javaファイルの変更があるもののみ記録
            if commit_dict['java_change']:
                commits.append(commit_dict)
        repo_dict['commits'] = commits
        all_commits.append(repo_dict)

    with open('data/sandbox/all_commits.json', mode='w', encoding='utf-8') as f_all_commits:
        json.dump(all_commits, f_all_commits, indent="\t")
            
    return


if __name__ == "__main__":
    main()