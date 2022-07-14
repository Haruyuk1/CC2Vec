import json
import pandas as pd


def main():
    json_source: str = "data/sandbox/spring_boot_detect.json"

    with open(json_source, mode="r", encoding="utf-8") as f:
        spring_boost_json = json.load(f)

    print(len(spring_boost_json["commits"]))

    class Commit:
        def __init__(self, commit) -> None:
            self.repository: str = commit["repository"]
            self.sha1: str = commit["sha1"]
            self.url: str = commit["url"]
            self.refactorings: list[dict] = commit["refactorings"]
        
        def __str__(self) -> str:
            res = f"repo:{self.repository}, sha1:{self.sha1[:6]}, url:{self.url}, refs:{len(self.refactorings)}"
            return res

    ref_commits: list[Commit] = []
    non_ref_commits: list[Commit] = []
    for c in spring_boost_json["commits"]:
        commit = Commit(c)
        if commit.refactorings:
            ref_commits.append(commit)
        else:
            non_ref_commits.append(commit)
    
    print(len(ref_commits))
    print(len(non_ref_commits))
            



    return


if __name__ == "__main__":
    main()