## PyPI Release Rules

When releasing a new version of `information-gain-aml`:

1. **Version sync**: Always update version in both `pyproject.toml` and `information_gain_aml/__init__.py`. They must match.
2. **Tests first**: Run `uv run pytest tests/ -v` and `uv run mypy information_gain_aml/` before building.
3. **Clean build**: Always `rm -rf dist/ build/ *.egg-info information_gain_aml.egg-info` before `python3 -m build`.
4. **Verify wheel**: Run `unzip -l dist/*.whl` and confirm `experiments/` and `environments/` are excluded.
5. **No re-uploads**: PyPI does not allow re-uploading the same version. If a release has a problem, bump the version.
6. **Semver**: PATCH for bug fixes, MINOR for new features, MAJOR for breaking API changes.
7. **Full procedure**: See `docs/releasing.md` for the complete step-by-step process.
