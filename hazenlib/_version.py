"""Single point place to retrieve the version."""

import importlib.metadata
import pathlib
import tomllib

distributions = importlib.metadata.packages_distributions().get("hazenlib", [])

__version__ = None
for dist_name in distributions:
    try:
        __version__ = importlib.metadata.version(dist_name)
        break
    except importlib.metadata.PackageNotFoundError:
        continue

if __version__ is None:
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
        __version__ = data["project"]["version"] + "+dev"
