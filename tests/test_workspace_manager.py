from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from trenni.bundle_repository import BundleRepositoryManager
from trenni.config import BundleConfig, TrenniConfig
from trenni.workspace_manager import WorkspaceManager


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_prepare_same_source_target_creates_pushable_tracking_branch(tmp_path):
    remote = tmp_path / "factorio.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        capture_output=True,
        text=True,
        check=True,
    )

    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "init", str(seed)],
        capture_output=True,
        text=True,
        check=True,
    )
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test User")
    (seed / "README.md").write_text("seed\n")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "checkout", "-b", "evolve")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "evolve")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/evolve")

    with patch.object(BundleRepositoryManager, "BUNDLES_DIR", tmp_path / "bundles"):
        bundle_repo = BundleRepositoryManager()
        bundle_repo.ensure_bare_clone("factorio", str(remote))

        config = TrenniConfig(
            workspace_root=str(tmp_path / "workspaces"),
            bundles={
                "factorio": BundleConfig.from_dict(
                    {
                        "source": {
                            "url": str(remote),
                            "evolve_selector": "evolve",
                        }
                    }
                )
            },
        )
        manager = WorkspaceManager(config, bundle_repo_manager=bundle_repo)

        target_source, worktree = manager._prepare_same_source_target(
            "job-123",
            "factorio",
            config.bundles["factorio"],
            "evolve",
        )

        assert target_source is not None
        assert worktree is not None
        assert target_source.resolved_ref
        assert _git(worktree, "branch", "--show-current") == "evolve"
        assert _git(worktree, "config", "--get", "branch.evolve.remote") == "origin"
        assert _git(worktree, "config", "--get", "branch.evolve.merge") == "refs/heads/evolve"

        (worktree / "CHANGELOG.md").write_text("ready\n")
        _git(worktree, "config", "user.email", "test@example.com")
        _git(worktree, "config", "user.name", "Test User")
        _git(worktree, "add", "CHANGELOG.md")
        _git(worktree, "commit", "-m", "publish")
        _git(worktree, "push")

        pushed_sha = _git(worktree, "rev-parse", "HEAD")
        remote_sha = _git(remote, "rev-parse", "refs/heads/evolve")
        assert remote_sha == pushed_sha

        manager.cleanup([worktree])
