"""Traçabilité : lecture du commit Git sans exécutable git (cas des conteneurs)."""

from src.common import md5_file, read_git_head

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_reads_commit_from_branch_reference(tmp_path):
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    assert read_git_head(tmp_path) == COMMIT


def test_reads_commit_from_packed_refs(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/develop\n")
    (tmp_path / ".git" / "packed-refs").write_text(f"# pack-refs\n{COMMIT} refs/heads/develop\n")
    assert read_git_head(tmp_path) == COMMIT


def test_detached_head_and_missing_repository(tmp_path):
    assert read_git_head(tmp_path) is None
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(COMMIT + "\n")
    assert read_git_head(tmp_path) == COMMIT


def test_md5_is_stable(tmp_path):
    fichier = tmp_path / "donnees.csv"
    fichier.write_bytes(b"date,consumption_gwh\n2017-01-01,1130.4\n")
    assert md5_file(fichier) == md5_file(fichier)
    assert len(md5_file(fichier)) == 32
