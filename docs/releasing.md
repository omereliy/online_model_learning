# Releasing to PyPI

## Package Info

| Field | Value |
|-------|-------|
| PyPI name | `information-gain-aml` |
| Import name | `information_gain_aml` |
| PyPI URL | https://pypi.org/project/information-gain-aml/ |
| TestPyPI URL | https://test.pypi.org/project/information-gain-aml/ |

## What Gets Published

Only `information_gain_aml/algorithms/` and `information_gain_aml/core/` are included in the wheel. The `experiments/` and `environments/` subpackages are excluded via `pyproject.toml` package discovery and stay in the repo for local development only.

## Prerequisites

- PyPI account with API token (https://pypi.org/manage/account/token/)
- `~/.pypirc` configured:
  ```ini
  [distutils]
  index-servers =
      pypi
      testpypi

  [pypi]
  username = __token__
  password = pypi-<your-token>

  [testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-<your-token>
  ```
- `chmod 600 ~/.pypirc`

## Automated Release (CI/CD)

Releases are automated via GitHub Actions. The workflow (`.github/workflows/publish.yml`) triggers on version tags and publishes to PyPI using [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) (OIDC — no API tokens needed).

### How to release

1. Update version in both `pyproject.toml` and `information_gain_aml/__init__.py`
2. Commit: `git commit -am "Bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin master --tags`
5. The workflow runs tests → builds → verifies wheel → waits for owner approval → publishes

Monitor progress at: https://github.com/omereliy/online_model_learning/actions

### One-time setup

These steps must be done once before the first automated release:

**1. Create `pypi` GitHub Environment:**
- Repo Settings → Environments → New environment → `pypi`
- Enable "Required reviewers" → add `omereliy`
- Deployment branches → restrict to `master`

**2. Configure PyPI Trusted Publisher:**
- https://pypi.org/manage/project/information-gain-aml/settings/publishing/
- Add GitHub Actions publisher: owner `omereliy`, repo `online_model_learning`, workflow `publish.yml`, environment `pypi`

**3. Branch protection on `master`:**
- Repo Settings → Branches → Add rule for `master`
- "Restrict who can push" → add only `omereliy`

---

## Manual Release Steps (fallback)

### 1. Bump the version

Update the version in **both** places (they must match):
- `pyproject.toml` → `version = "X.Y.Z"`
- `information_gain_aml/__init__.py` → `__version__ = "X.Y.Z"`

### 2. Run tests

```bash
uv run pytest tests/ -v
uv run mypy information_gain_aml/
```

### 3. Clean and build

```bash
rm -rf dist/ build/ *.egg-info information_gain_aml.egg-info
python3 -m build
```

### 4. Verify wheel contents

```bash
unzip -l dist/information_gain_aml-*.whl
```

Confirm:
- `information_gain_aml/algorithms/` is present
- `information_gain_aml/core/` is present
- `information_gain_aml/experiments/` is **NOT** present
- `information_gain_aml/environments/` is **NOT** present

### 5. Upload to TestPyPI (optional but recommended for new versions)

```bash
python3 -m twine upload --repository testpypi dist/*
```

Test install in a clean environment:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ information-gain-aml==X.Y.Z
python3 -c "from information_gain_aml.algorithms.information_gain import InformationGainLearner; print('OK')"
```

### 6. Upload to PyPI

```bash
python3 -m twine upload dist/*
```

### 7. Verify

```bash
pip install information-gain-aml==X.Y.Z
python3 -c "import information_gain_aml; print(information_gain_aml.__version__)"
```

## Versioning

Use semantic versioning (`MAJOR.MINOR.PATCH`):
- **PATCH** (0.1.0 → 0.1.1): Bug fixes, internal changes that don't affect the public API
- **MINOR** (0.1.0 → 0.2.0): New features, non-breaking API additions
- **MAJOR** (0.1.0 → 1.0.0): Breaking API changes (e.g., changing `InformationGainLearner` method signatures)

## Dependencies

Core dependencies (shipped with the package):
- `python-sat` — SAT solver for CNF management
- `unified-planning` — PDDL parsing and domain representation
- `typing-extensions` — Backported type hints

Optional dependencies (for local experiment running):
```bash
pip install "information-gain-aml[experiments]"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `twine` not found | `uv pip install build twine` |
| 403 on upload | Check `~/.pypirc` token is valid |
| Version already exists | You cannot re-upload the same version to PyPI. Bump the version. |
| Wheel includes `experiments/` | Check `pyproject.toml` `[tool.setuptools.packages.find]` include list |
