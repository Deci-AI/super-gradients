from typing import List
import git


class GitHelper:
    def __init__(self, git_path: str):

        self.repo = git.Repo(git_path)

    def diff_files(self, source_branch: str, current_branch: str) -> List[str]:
        source_commit = self.repo.commit(source_branch)
        current_commit = self.repo.commit(current_branch)
        return [diff.a_path for diff in source_commit.diff(current_commit) if ".py" in diff.a_path]

    def load_branch_file(self, branch: str, file_path: str) -> str:
        tree = self.repo.commit(branch).tree

        try:  # It looks like there is no simple way to check if a file exists in the tree... So we directly check with try/except
            return tree[file_path].data_stream.read()
        except KeyError:
            return ""
