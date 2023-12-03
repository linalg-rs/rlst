try:
    import tomllib
except ImportError:
    import tomli as tomllib
import os
import pytest


def walk_dirs(dir, file):
    out = []
    if os.path.isfile(os.path.join(dir, file)):
        out.append(os.path.join(dir, file))
    for f in os.listdir(dir):
        if not f.startswith(".") and os.path.isdir(os.path.join(dir, f)):
            out += walk_dirs(os.path.join(dir, f), file)
    return out


def test_dependencies():
    root_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..")
    if not os.path.isfile(os.path.join(root_dir, "Cargo.toml")):
        pytest.skip("Test must be run from source folder")

    cargos = walk_dirs(root_dir, "Cargo.toml")

    deps = {}
    errors = []
    for c in cargos:
        with open(c, "rb") as f:
            data = tomllib.load(f)
            if "dependencies" in data:
                for d, info in data["dependencies"].items():
                    if isinstance(info, dict):
                        if "version" not in info:
                            info = None
                        else:
                            info = info["version"]
                    if d not in deps:
                        deps[d] = (info, c)
                    elif deps[d][0] != info:
                        errors.append(
                            f"Version of {d} in {c} ({info}) does not agree "
                            f"with version in {deps[d][1]} ({deps[d][0]})")
    if len(errors) > 0:
        raise ValueError("\n".join(errors))
