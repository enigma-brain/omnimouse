import git
import os
from typing import Dict, Any, Dict, Tuple
from pathlib import Path
from subprocess import check_output, run

from omnimouse.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_git_info(repo_dir: str | os.PathLike) -> Tuple[Dict[str, Any], str | None]:
    """Get git information using GitPython library.
    
    Args:
        logger: WandbLogger or MLFlowLogger instance (or None)
        
    Returns:
        Dictionary containing git information for hyperparameter logging
    """
    git_info, git_diff = {}, None
    
    try:
        repo_path = Path(repo_dir)

        # Initialize repo object
        repo = git.Repo(repo_dir)
        
        # Get current git commit hash
        git_hash = repo.head.commit.hexsha
        git_info["git/commit_hash"] = git_hash
        
        # Get short hash for display
        git_hash_short = repo.head.commit.hexsha[:7]
        git_info["git/commit_hash_short"] = git_hash_short
        
        # Check if tracked files are modified (excluding untracked files)
        is_dirty = repo.is_dirty(untracked_files=False)
        git_info["git/is_dirty"] = is_dirty
        
        # Check if untracked files exist
        exists_untracked = len(repo.untracked_files) > 0
        git_info["git/exists_untracked"] = exists_untracked
        git_info["git/untracked_files"] = repo.untracked_files
        
        # Get current branch name
        try:
            git_branch = repo.active_branch.name
            git_info["git/branch"] = git_branch
        except Exception:
            # Handle detached HEAD state
            git_info["git/branch"] = "detached"
        
        # Only create diff if tracked files are dirty (not for untracked files)
        if is_dirty:
            # Get both staged and unstaged changes for tracked files only
            git_diff = repo.git.diff('HEAD')

    except Exception as e:
        log.warning(f"Failed to get git information for repo {repo_dir}: {e}")
        git_info["git/commit_hash"] = "unknown"
        git_info["git/commit_hash_short"] = "unknown"
        git_info["git/is_dirty"] = None
        git_info["git/exists_untracked"] = None
        git_info["git/branch"] = "unknown"
    
    return git_info, git_diff

def get_git_filtered_files(code_dir: str):
    """
    Get all files in a directory that aren't ignored by git or in .git folder.
    
    Returns:
        List of (absolute_path, relative_path) tuples
    """
    # get .git folder path
    git_dir_path = Path(
        check_output(["git", "rev-parse", "--git-dir"])
        .strip()
        .decode("utf8")
    ).resolve()
    
    filtered_files = []
    for path in Path(code_dir).resolve().rglob("*"):
        # don't upload files ignored by git
        command = ["git", "check-ignore", "-q", str(path)]
        not_ignored = run(command).returncode == 1
        
        # don't upload files from .git folder
        not_git = not str(path).startswith(str(git_dir_path))
        
        if path.is_file() and not_git and not_ignored:
            relative_path = path.relative_to(code_dir)
            filtered_files.append((path, relative_path))
    
    return filtered_files