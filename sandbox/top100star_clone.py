import os
import requests
import json
import git
import pprint
from multiprocessing import Pool


github_repository_dir = "repos/"


def getTop100StarJavaProject():
    header = {"Accept": "application/vnd.github.mercy-preview+json"}
    URL = "https://api.github.com/search/repositories?q=stars:%3E1+language:Java&s=stars&type=Repositories&per_page=100"
    r = requests.get(URL, headers=header)
    json_text = json.loads(r.text)
    return json_text


def cloneFromProject(project):
        try:
            clone_url, name = project['clone_url'], project['name']
            if os.path.exists(github_repository_dir + name): # don't clone if already exists
                print(f"{name} already exists, so skipped.")
                return
            git.Repo.clone_from(clone_url, github_repository_dir + name)
            print(f"{name} cloned successfully.")
        except Exception as e:
            print(e)


def main():
    response = getTop100StarJavaProject()
    java_projects = response["items"]

    # filter non-english project
    java_projects = list(filter(lambda e: e['description'].isascii(), java_projects))

    # clone with multiprocess
    p = Pool(10)
    p.map(cloneFromProject, java_projects)

    return


if __name__ == "__main__":
    main()