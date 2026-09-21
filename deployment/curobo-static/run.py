"""Launch the explicitly isolated static probe; never launch ROS or a driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVIEWED_IMAGE_ID = "sha256:b34c1bcf9fc094ecb39e7de3591f4e22112b9713e2892783259a2ac0cc7d949e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--wrapper-checkout", type=Path, required=True)
    parser.add_argument("--gpu-cache", type=Path, required=True)
    parser.add_argument("--image", default="rammp-adl-curobo-static:jp6-v078")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.gpu_cache.mkdir(parents=True, exist_ok=True)
    image = json.loads(subprocess.run(["docker", "image", "inspect", args.image], check=True, capture_output=True, text=True).stdout)[0]
    if image["Id"] != REVIEWED_IMAGE_ID:
        raise RuntimeError("Image differs from the reviewed static image; review its provenance before updating the pin")
    if image["Config"]["Entrypoint"] != ["python3"] or image["Config"].get("Volumes"):
        raise RuntimeError("Unexpected image entrypoint; review the selected image")
    name = "rammp-adl-static-" + uuid.uuid4().hex[:12]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--runtime", "nvidia",
               "--read-only", "--tmpfs", "/tmp:rw,size=2g", "--entrypoint", "python3",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/workspace:/opt/pinned-wrapper/core",
               "-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory", "-e", "GIT_CONFIG_VALUE_0=/opt/pinned-wrapper"]
    bindings = [(args.wrapper_checkout.resolve(), "/opt/pinned-wrapper", "ro"), (HERE, "/probe", "ro"),
                (args.output, "/out", "rw"), (args.gpu_cache.resolve(), "/root/.cache", "rw")]
    bindings.extend((ROOT / name, "/workspace/" + name, "ro") for name in ("rammp_adl", "skills", "schemas", "config", "simulation"))
    bindings.append((ROOT / "tools/check_design.py", "/workspace/tools/check_design.py", "ro"))
    for source, target, mode in bindings:
        if not source.exists():
            raise FileNotFoundError(source)
        command.extend(["-v", str(source) + ":" + target + ":" + mode])
    command.extend([image["Id"], "/probe/run_static.py", "--wrapper-source", "/opt/pinned-wrapper", "--output-dir", "/out/probe"])
    (args.output / "invocation.json").write_text(json.dumps({"image_id": image["Id"], "command": command,
                                                           "hardware_commands": False, "network": "none"}, indent=2) + "\n")
    try:
        with (args.output / "probe.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True)
        raise
    finally:
        # The GPU process runs as root inside its isolated image. Return only
        # this newly created artifact directory to the invoking local user.
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--read-only", "--entrypoint", "python3",
                        "-v", str(args.output) + ":/out", image["Id"], "-c",
                        "import os,pathlib,sys; root=pathlib.Path('/out'); "
                        "[os.chown(p,int(sys.argv[1]),int(sys.argv[2]),follow_symlinks=False) "
                        "for p in [root,*root.rglob('*')]]", str(os.getuid()), str(os.getgid())], check=True)
    print(json.dumps({"exit_code": result.returncode, "output": str(args.output), "hardware_commands": False}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
