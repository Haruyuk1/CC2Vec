import git


def main():
    repo_name = 'repos/PhotoView'
    sha1 = '8a592bf2ad78d2f5330e293b937e8e1e8fa347b9'

    repo = git.Repo(repo_name)
    commit = repo.commit(sha1)

    count = 0
    for diff in commit.diff(commit.parents[0], create_patch=True):
        print(diff)
        count += 1
    print(count)

    return


if __name__ == "__main__":
    main()