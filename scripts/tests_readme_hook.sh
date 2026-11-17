#!/usr/bin/env bash
# The pre-commit hook of tests/README.md: run scripts/make_tests_readme.py with a Python
# that has pytest and multigeom (it collects the tests). Commits made from an editor
# often do not see the active conda environment, so the Python is looked up in order:
#
#   1. the one set for this clone:  git config tests-readme.python /path/to/python
#   2. the environment of the README:  ./env/bin/python
#   3. python, then python3, on the PATH
#
# If none has pytest and multigeom, the hook warns and lets the commit through: GitHub
# Actions still checks the README (python scripts/make_tests_readme.py --check).

candidates=("$(git config --get tests-readme.python)" ./env/bin/python python python3)

for candidate in "${candidates[@]}"; do
    # skip empty entries and Pythons without what the script needs
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c "import pytest, multigeom" >/dev/null 2>&1 || continue

    exec "$candidate" scripts/make_tests_readme.py
done

echo "tests-readme: no Python with pytest and multigeom found; README not checked." >&2
echo "  Set one with: git config tests-readme.python /path/to/python" >&2
exit 0
