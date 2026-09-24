#!/usr/bin/env python3
"""Put the adl node back into the sheppy deployment manifest after a deployment update removed it.

    python tools/sheppy_adl_node.py status              is the adl node in the manifest, armed or not
    python tools/sheppy_adl_node.py install [--armed]    add it (or replace it) from deployment/sheppy/adl-node.yaml

The node definition lives in this repository; the manifest belongs to the deployment and is updated from
upstream, which does not carry this node. `--armed` sets commissioned and arm_motion: the operator's decision
that this node may move the arm. A backup of the manifest is written beside it first.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT = ROOT/"deployment/sheppy/adl-node.yaml"
MANIFEST = Path("/home/abra/rammp-deployments/december_2026/sheppy-manifest.yaml")


def node(armed):
    return yaml.safe_load(FRAGMENT.read_text().replace("@ARMED@", "true" if armed else "false"))


def status(args):
    nodes = {n["name"]: n for n in yaml.safe_load(Path(args.manifest).read_text())["nodes"]}
    if "adl" not in nodes:
        print("adl node: missing (run install)")
        return
    command = nodes["adl"]["alternatives"][0].get("command", "")
    print("adl node: present;", "armed" if "arm_motion:=true" in command else "not armed", ";",
          "follows active-root" if "active-root" in command else "fixed checkout")


def install(args):
    path = Path(args.manifest)
    text = path.read_text()
    document = yaml.safe_load(text)
    backup = path.with_name(path.name+f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
    shutil.copy2(path, backup)
    lines = text.rstrip("\n").splitlines()
    names = [n["name"] for n in document["nodes"]]
    if "adl" in names:
        # Drop the existing block: from its "- name: adl" line to the next top-level node or the end.
        start = next(i for i, line in enumerate(lines) if line.strip() == "- name: adl")
        end = next((i for i in range(start+1, len(lines)) if lines[i].startswith("  - name: ")), len(lines))
        lines = lines[:start]+lines[end:]
    block = yaml.safe_dump([node(args.armed)], sort_keys=False, width=100000, default_flow_style=False)
    indented = ["  "+line if line else line for line in block.rstrip("\n").splitlines()]
    path.write_text("\n".join(lines+["  # LOCAL EDIT (do not merge): RAMMP-generalized's adl node, from "
                                     "RAMMP-generalized/deployment/sheppy/adl-node.yaml"]+indented)+"\n")
    check = {n["name"]: n for n in yaml.safe_load(path.read_text())["nodes"]}
    assert "adl" in check and len(check) == len(set(names) | {"adl"}), "manifest did not round-trip"
    print(f"adl node installed ({'armed' if args.armed else 'not armed'}); backup at {backup}")
    print("load it with: cd /home/abra/rammp-deployments/december_2026 && sheppy up adl --manifest sheppy-manifest.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(MANIFEST))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status").set_defaults(function=status)
    add = commands.add_parser("install")
    add.add_argument("--armed", action="store_true")
    add.set_defaults(function=install)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    sys.exit(main())
