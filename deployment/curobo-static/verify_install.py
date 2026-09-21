"""Read-only source/binary provenance verification for the derived image."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

if not __debug__:
    raise RuntimeError("Provenance verification requires assertions enabled; Python -O is unsupported")

PIN = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ROOT = Path("/opt/curobo")


def git(*arguments):
    return subprocess.run(["git", "-c", "safe.directory=/opt/curobo", "-C", str(ROOT), *arguments],
                          check=True, capture_output=True, text=True).stdout.strip()


def verify():
    import curobo
    assert git("rev-parse", "HEAD") == PIN, "cuRobo source revision changed"
    assert not git("status", "--porcelain", "--untracked-files=no"), "cuRobo tracked source modified"
    installed = Path(curobo.__file__).resolve().parent
    hashes = {}
    absent = []
    for relative in git("ls-files", "src/curobo").splitlines():
        source = ROOT / relative
        target = installed / source.relative_to(ROOT / "src/curobo")
        if not target.is_file():
            assert source.suffix == ".h", "Non-header source omitted from package: " + relative
            absent.append(relative)
            continue
        assert target.read_bytes() == source.read_bytes(), "Installed source differs: " + relative
        hashes[str(target.relative_to(installed))] = hashlib.sha256(target.read_bytes()).hexdigest()
    binaries = {}
    for target in sorted(installed.rglob("*.so")):
        candidates = list((ROOT / "build").rglob(target.name))
        assert len(candidates) == 1 and candidates[0].read_bytes() == target.read_bytes(), "Binary/build mismatch"
        binaries[str(target.relative_to(installed))] = hashlib.sha256(target.read_bytes()).hexdigest()
    assert len(binaries) == 5, "Unexpected CUDA binary set"
    return {"curobo_commit": PIN, "source_clean": True, "module_version": curobo.__version__,
            "distribution_version": metadata.version("nvidia_curobo"), "installed_package": str(installed),
            "installed_source_sha256": hashes, "omitted_source_headers": absent,
            "installed_binary_sha256": binaries,
            "package_versions": {name: metadata.version(name) for name in
                                 ("torch", "numpy", "warp-lang", "setuptools", "setuptools_scm")},
            "gpu_planning_tested": False, "hardware_commands": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--before", type=Path)
    group.add_argument("--after", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.after:
        assert report["module_version"] == report["distribution_version"] == "0.7.8"
        import mujoco
        report["mujoco_version"] = mujoco.__version__
    target = args.after or args.before
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("sha256")}))
