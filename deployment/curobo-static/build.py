"""Build a separate static-planning image from verified local inputs, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE_IMAGE = "rammp-curobo:jp6"
BASE_ID = "sha256:081953f22faa1f815196ea1726a717ea73a690449ef8175e2949aec4629db28d"
WRAPPER_PIN = "320872b709b276fc7283190d24edef7f8632bec9"
HERE = Path(__file__).resolve().parent


def output(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True, help="New build context directory")
    parser.add_argument("--wrapper-repository", type=Path, required=True)
    parser.add_argument("--wrapper-checkout", type=Path, required=True, help="New isolated Git checkout directory")
    parser.add_argument("--image", default="rammp-adl-curobo-static:jp6-v078")
    args = parser.parse_args()
    existing = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True)
    if existing.returncode == 0:
        raise FileExistsError("Derived image tag already exists; use a new tag")
    if output("docker", "image", "inspect", BASE_IMAGE, "--format", "{{.Id}}") != BASE_ID:
        raise RuntimeError("Local base image identity changed; review before rebuilding")
    expected = {line.split("--hash=sha256:")[1].strip() for line in (HERE / "wheels.lock").read_text().splitlines() if line.strip()}
    found = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in args.wheel_dir.glob("*.whl")}
    if not expected <= found.keys():
        raise RuntimeError("Pinned offline dependency wheels missing or changed")
    if args.context.exists() or args.wrapper_checkout.exists():
        raise FileExistsError("Use new context and wrapper-checkout directories")
    if output("git", "-C", str(args.wrapper_repository), "cat-file", "-t", WRAPPER_PIN) != "commit":
        raise RuntimeError("Pinned wrapper revision unavailable locally")
    args.context.mkdir(parents=True, exist_ok=False)
    (args.context / "wheels").mkdir()
    for name in ("Dockerfile", "verify_install.py", "wheels.lock"):
        shutil.copyfile(HERE / name, args.context / name)
    for value in expected:
        shutil.copyfile(found[value], args.context / "wheels" / found[value].name)
    subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", str(args.wrapper_repository), str(args.wrapper_checkout)], check=True)
    subprocess.run(["git", "-C", str(args.wrapper_checkout), "checkout", "--detach", WRAPPER_PIN], check=True)
    with (args.context / "build.log").open("w") as log:
        subprocess.run(["docker", "build", "--network", "none", "--pull=false", "-t", args.image, str(args.context)],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    report = {"base_image": BASE_IMAGE, "base_id": BASE_ID, "image": args.image,
              "image_id": output("docker", "image", "inspect", args.image, "--format", "{{.Id}}"),
              "wrapper_commit": WRAPPER_PIN, "wrapper_checkout": str(args.wrapper_checkout.resolve()),
              "wheel_sha256": sorted(expected), "hardware_commands": False}
    (args.context / "build-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
