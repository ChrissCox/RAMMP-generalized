#!/usr/bin/env python3
"""Build instrumented upstream driver; compilation never starts robot control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from prepare import prepare


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sdk(sdk: Path) -> dict:
    header = sdk / "include/client/RouterClient.h"
    library = sdk / "lib/release/libKortexApiCpp.a"
    if not header.is_file() or not library.is_file():
        raise ValueError("Expected Kortex C++ include/client/RouterClient.h and lib/release/libKortexApiCpp.a")
    members = subprocess.check_output(["ar", "t", str(library)], text=True).splitlines()
    if not members:
        raise ValueError("Kortex archive is empty")
    # Inspect every archive member's ELF machine, not just its directory name.
    with tempfile.TemporaryDirectory(prefix="rammp-sdk-inspect-") as temporary:
        subprocess.run(["ar", "x", str(library)], cwd=temporary, check=True)
        objects = list(Path(temporary).iterdir())
        if not objects:
            raise ValueError("No objects extracted from Kortex archive")
        for obj in objects:
            with obj.open("rb") as stream:
                data = stream.read(20)
            if data[:4] != b"\x7fELF" or data[5] != 1 or struct.unpack("<H", data[18:20])[0] != 183:
                raise ValueError("Kortex SDK contains a non-AArch64 ELF object")
    files = {str(path.relative_to(sdk)): sha256(path)
             for path in sorted((sdk / "include").rglob("*")) if path.is_file()}
    files["lib/release/libKortexApiCpp.a"] = sha256(library)
    return {"root": str(sdk), "library_sha256": sha256(library),
            "file_manifest_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "elf_machine": "AArch64", "archive_members": len(members)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "hardware"), default="sim")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/rammp-driver-hardware-build"))
    parser.add_argument("--kortex-sdk", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "hardware" and args.kortex_sdk is None:
        parser.error("--mode hardware requires --kortex-sdk; this builds only and never launches")
    if args.mode == "sim" and args.kortex_sdk is not None:
        parser.error("SimTransport builds do not consume a Kortex SDK")
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8")
    workspace = args.workspace.resolve()
    manifest = prepare(workspace)
    manifest["kortex_compiled"] = args.mode == "hardware"
    manifest["sdk"] = inspect_sdk(args.kortex_sdk.resolve()) if args.kortex_sdk else None
    script = '''set -e
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.zsh
export CMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/cmeel.prefix:$CMAKE_PREFIX_PATH
export MAKEFLAGS="-j$4"
cd "$1"
colcon build --base-paths "$1/src" --build-base "$1/build" --install-base "$1/install" \
  --packages-up-to kinova_gen3_ros2 --parallel-workers 2 --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  "-DKINOVA_ENABLE_KORTEX=$2" "-DKORTEX_HW_DIR=$3"
'''
    with (workspace / "build.log").open("w") as log:
        result = subprocess.run(["zsh", "-fc", script, "rammp-driver-build", str(workspace),
                                 "ON" if args.mode == "hardware" else "OFF",
                                 str(args.kortex_sdk.resolve()) if args.kortex_sdk else "",
                                 str(args.jobs)], stdout=log, stderr=subprocess.STDOUT)
    manifest["build_returncode"] = result.returncode
    binary = workspace / "install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node"
    if result.returncode == 0:
        manifest["executable"] = str(binary)
        manifest["executable_sha256"] = sha256(binary)
    (workspace / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
