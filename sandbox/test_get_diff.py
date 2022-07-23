import re
import git
import javalang
from pyparsing import line_start


def main():
    repo_name = 'repos/PhotoView'
    sha1 = '8a592bf2ad78d2f5330e293b937e8e1e8fa347b9'

    repo = git.Repo(repo_name)
    commit = repo.commit(sha1)

    RE_HUNK_PATTERN = r'''
        @@\s-([0-9]*?),([0-9]*?)\s\+([0-9]*?),([0-9]*?)\s@@.*?\n # 1行目
        (.*?) # 2行目以降
        (?=@@|$)'''

    count = 0
    for diff in commit.diff(commit.parents[0], create_patch=True):
        count += 1
        diff: git.Diff = diff
        unified_diff = diff.diff.decode('utf-8')
        matches = re.findall(RE_HUNK_PATTERN, unified_diff, flags=re.DOTALL+re.VERBOSE)
        
        a_blob_tokens = []
        b_blob_tokens = []
        if diff.a_blob and diff.a_blob.name.endswith('.java'):
            code = diff.a_blob.data_stream.read().decode('utf-8')
            a_blob_tokens = list(javalang.tokenizer.tokenize(code))
        if diff.b_blob and diff.b_blob.name.endswith('.java'):
            code = diff.b_blob.data_stream.read().decode('utf-8')
            tokens = list(javalang.tokenizer.tokenize(code))
            b_blob_tokens = list(javalang.tokenizer.tokenize(code))
        
        print(unified_diff)

        for match in matches:
            hunk_removed = []
            hunk_added = []
            b_row, a_row = int(match[0]), int(match[2])
            diff_lines: list[str] = match[4].split('\n')
            for line in diff_lines:
                if line.startswith('-'):
                    hunk_removed.append(b_row)
                    b_row += 1
                elif line.startswith('+'):
                    hunk_added.append(a_row)
                    a_row += 1
                else:
                    b_row += 1
                    a_row += 1

            if a_blob_tokens:
                filtered_a_blob_tokens = [token for token in a_blob_tokens if token.position[0] in hunk_added]
                print(filtered_a_blob_tokens)
            if b_blob_tokens:
                filtered_b_blob_tokens = [token for token in b_blob_tokens if token.position[0] in hunk_removed]
                print(filtered_b_blob_tokens)

            print('removed', hunk_removed)
            print('added', hunk_added)

        
    


    print(count)

    return


if __name__ == "__main__":
    main()