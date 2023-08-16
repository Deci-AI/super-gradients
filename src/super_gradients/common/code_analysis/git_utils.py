from typing import List

try:
    import git
except ImportError:
    raise ImportError("The 'git' library is required but not found. Please install the `gitpython` with version as specified in `requirements.dev.txt`.")


class GitHelper:
    """A helper class to interact with a (local) Git repository."""

    def __init__(self, git_path: str):
        """
        :param git_path: Path to the Git repository.
        """

        self.repo = git.Repo(git_path)

    def diff_files(self, source_branch: str, current_branch: str) -> List[str]:
        """Get the differences in files between the source branch and the current branch, only considering '.py' files.
        :param source_branch: The source branch for comparison.
        :param current_branch: The current branch for comparison.
        :return:               List of file paths that have changed between the two branches, considering only '.py' files.
        """

        source_commit = self.repo.commit(source_branch)
        current_commit = self.repo.commit(current_branch)
        return [diff.a_path for diff in source_commit.diff(current_commit) if ".py" in diff.a_path]

    def load_branch_file(self, branch: str, file_path: str) -> str:
        """Load the contents of a file from a specific branch. Return an empty string if the file does not exist.
        :param branch:    The branch from which to load the file.
        :param file_path: The path of the file within the repository.
        :return:          The content of the file as a string or an empty string if the file does not exist.
        """

        tree = self.repo.commit(branch).tree

        try:  # It looks like there is no simple way to check if a file exists in the tree... So we directly check with try/except
            return tree[file_path].data_stream.read()
        except KeyError:
            return ""
