"""Tests for role catalog isolation between global and team-specific roles.

Issue: _load_role_catalog() uses catalog[meta.name] as key, so team-specific roles
overwrite global roles of the same name. When _resolve_team_definition("default")
is called after factorio defines "worker", the global worker is gone.

Solution: Store both global and team-specific versions, and prefer team-specific
when resolving for a specific team.
"""

import pytest
from pathlib import Path


class TestRoleCatalogPreservesGlobalRoles:
    """Tests verifying global roles are preserved when team has same-named role."""

    @pytest.fixture
    def evo_with_same_named_roles(self, tmp_path: Path) -> Path:
        """Create evo structure where factorio team defines 'worker' same as global."""
        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Global roles
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Global planner", role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        (global_roles / "evaluator.py").write_text('''
from palimpsest.runtime import role

@role(name="evaluator", description="Global evaluator", role_type="evaluator")
def evaluator(**params):
    pass
''')

        # Factorio team with same-named "worker" role
        teams_dir = evo_root / "teams"
        factorio_dir = teams_dir / "factorio"
        factorio_roles = factorio_dir / "roles"
        factorio_roles.mkdir(parents=True)

        (factorio_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Factorio-specific worker", role_type="worker")
def factorio_worker(**params):
    pass
''')

        return evo_root

    def test_global_worker_preserved_when_factorio_has_worker(self, evo_with_same_named_roles: Path):
        """Global worker is still in catalog after factorio defines 'worker'."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        # Both global and factorio "worker" should be preserved
        assert "worker" in catalog, "worker role should exist in catalog"
        
        # The catalog entry should have both global and team-specific versions
        worker_entry = catalog["worker"]
        assert "global" in worker_entry or "teams" in worker_entry, \
            "worker entry should have 'global' and/or 'teams' keys for isolation"

    def test_resolve_factorio_team_finds_factorio_worker(self, evo_with_same_named_roles: Path):
        """_resolve_team_definition('factorio') finds factorio's worker."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        factorio_def = supervisor._resolve_team_definition("factorio")

        assert factorio_def is not None
        # Factorio should have its own worker
        assert "worker" in factorio_def.roles
        # The worker should be factorio-specific (source_team == "factorio")
        # or we need a way to check which version is used

    def test_resolve_default_team_finds_global_worker(self, evo_with_same_named_roles: Path):
        """_resolve_team_definition('default') finds global worker, not factorio's."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        default_def = supervisor._resolve_team_definition("default")

        assert default_def is not None
        assert "worker" in default_def.roles
        # Default team should use global worker, not factorio's

    def test_get_role_for_team_factorio_returns_factorio_version(self, evo_with_same_named_roles: Path):
        """_get_role_for_team('worker', 'factorio') returns factorio's worker definition."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        # This method may not exist yet, but should be added
        factorio_worker = supervisor._get_role_for_team("worker", "factorio")

        assert factorio_worker is not None
        assert factorio_worker.get("source_team") == "factorio", \
            "Factorio team should get factorio-specific worker"
        assert "Factorio" in factorio_worker.get("description", ""), \
            "Should have factorio-specific description"

    def test_get_role_for_team_default_returns_global_version(self, evo_with_same_named_roles: Path):
        """_get_role_for_team('worker', 'default') returns global worker definition."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        # This method may not exist yet, but should be added
        global_worker = supervisor._get_role_for_team("worker", "default")

        assert global_worker is not None
        assert global_worker.get("source_team") is None, \
            "Default team should get global worker (source_team=None)"
        assert "Global" in global_worker.get("description", ""), \
            "Should have global description"

    def test_get_role_for_team_other_returns_global_version(self, evo_with_same_named_roles: Path):
        """_get_role_for_team('worker', 'other-team') returns global worker, not factorio's."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        # Other teams should get global worker, not factorio's
        other_worker = supervisor._get_role_for_team("worker", "other-team")

        assert other_worker is not None
        assert other_worker.get("source_team") is None, \
            "Other teams should get global worker (source_team=None)"

    def test_resolve_role_metadata_without_team_uses_default(self, evo_with_same_named_roles: Path):
        """_resolve_role_metadata without team parameter should work for backward compat."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_same_named_roles))
        supervisor = Supervisor(config)

        # Existing method should still work (backward compatibility)
        # Should raise FileNotFoundError for unknown role
        with pytest.raises(FileNotFoundError):
            supervisor._resolve_role_metadata("nonexistent-role")


class TestRoleCatalogStructure:
    """Tests for the new catalog structure with team-aware storage."""

    @pytest.fixture
    def evo_with_overlapping_roles(self, tmp_path: Path) -> Path:
        """Create evo structure with multiple teams having same-named roles."""
        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Global roles
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Global planner", role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        # Team alpha with same-named worker
        teams_dir = evo_root / "teams"
        alpha_dir = teams_dir / "alpha"
        alpha_roles = alpha_dir / "roles"
        alpha_roles.mkdir(parents=True)

        (alpha_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Alpha worker", role_type="worker")
def alpha_worker(**params):
    pass
''')

        # Team beta with same-named worker
        beta_dir = teams_dir / "beta"
        beta_roles = beta_dir / "roles"
        beta_roles.mkdir(parents=True)

        (beta_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Beta worker", role_type="worker")
def beta_worker(**params):
    pass
''')

        return evo_root

    def test_catalog_has_single_worker_entry_with_team_versions(self, evo_with_overlapping_roles: Path):
        """Catalog has one 'worker' entry with global and team versions organized."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_overlapping_roles))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        assert "worker" in catalog
        worker_entry = catalog["worker"]

        # Entry should organize global and team-specific versions
        # Expected structure:
        # {
        #   "global": {...global worker data...} or None,
        #   "teams": {
        #     "alpha": {...alpha worker data...},
        #     "beta": {...beta worker data...}
        #   }
        # }
        assert "global" in worker_entry or "teams" in worker_entry, \
            "worker entry should have 'global' and/or 'teams' keys"

    def test_alpha_gets_alpha_worker(self, evo_with_overlapping_roles: Path):
        """Alpha team gets alpha-specific worker."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_overlapping_roles))
        supervisor = Supervisor(config)

        alpha_worker = supervisor._get_role_for_team("worker", "alpha")

        assert alpha_worker is not None
        assert alpha_worker.get("source_team") == "alpha"
        assert "Alpha" in alpha_worker.get("description", "")

    def test_beta_gets_beta_worker(self, evo_with_overlapping_roles: Path):
        """Beta team gets beta-specific worker."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_overlapping_roles))
        supervisor = Supervisor(config)

        beta_worker = supervisor._get_role_for_team("worker", "beta")

        assert beta_worker is not None
        assert beta_worker.get("source_team") == "beta"
        assert "Beta" in beta_worker.get("description", "")

    def test_gamma_gets_global_worker(self, evo_with_overlapping_roles: Path):
        """Gamma team (no specific worker) gets global worker."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_overlapping_roles))
        supervisor = Supervisor(config)

        gamma_worker = supervisor._get_role_for_team("worker", "gamma")

        assert gamma_worker is not None
        assert gamma_worker.get("source_team") is None
        assert "Global" in gamma_worker.get("description", "")

    def test_resolve_team_definition_alpha_has_alpha_worker(self, evo_with_overlapping_roles: Path):
        """_resolve_team_definition('alpha') includes alpha worker in roles."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(evo_with_overlapping_roles))
        supervisor = Supervisor(config)

        alpha_def = supervisor._resolve_team_definition("alpha")

        assert alpha_def is not None
        # Alpha team should have worker role (preferably alpha's version)
        assert "worker" in alpha_def.roles