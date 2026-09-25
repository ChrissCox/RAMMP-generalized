"""The skill library: every version of every learned skill, its description and its record.

On disk under artifacts/skills/<name>/: v<N>.py (the source, never edited), and skill.json (description,
versions, each version's successes, failures and evidence, and its status). A version is provisional when
written, verified after promote_after verified runs, retired after retire_after failures in a row.
Search is by the words of the description for now; embeddings come with local perception.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from .gate import check_source

NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class SkillLibrary:
    def __init__(self, root, *, promote_after=3, retire_after=3):
        self.root = Path(root)
        self.promote_after, self.retire_after = promote_after, retire_after

    def _record(self, name):
        path = self.root/name/"skill.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def _write(self, name, record):
        path = self.root/name/"skill.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1)+"\n")
        tmp.replace(path)

    def save(self, name, source, *, task="", parent=None, author="model"):
        """A new version of a skill; the source must pass the gate. Returns the version number."""
        if not NAME.match(name):
            raise ValueError("a skill name is lowercase words joined by underscores")
        description = check_source(source)
        (self.root/name).mkdir(parents=True, exist_ok=True)
        record = self._record(name) or {"name": name, "versions": []}
        version = len(record["versions"])+1
        (self.root/name/f"v{version}.py").write_text(source)
        record["description"] = description
        record["versions"].append({"version": version, "status": "provisional", "task": task, "parent": parent,
                                   "author": author, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "successes": 0, "failures": 0, "failures_in_a_row": 0, "evidence": []})
        self._write(name, record)
        return version

    def load(self, name, version=None):
        """The version asked for, else the newest verified, else the newest provisional; retired ones never."""
        record = self._record(name)
        if record is None:
            raise KeyError(f"no skill {name}")
        versions = [v for v in record["versions"] if v["status"] != "retired"]
        if version is not None:
            versions = [v for v in record["versions"] if v["version"] == version]
        elif any(v["status"] == "verified" for v in versions):
            versions = [v for v in versions if v["status"] == "verified"]
        if not versions:
            raise KeyError(f"no usable version of {name}")
        chosen = versions[-1]
        return {"name": name, "version": chosen["version"], "status": chosen["status"], "description": record["description"],
                "source": (self.root/name/f"v{chosen['version']}.py").read_text()}

    def record_outcome(self, name, version, *, verified, evidence=None):
        """A run's outcome, by measurement: promotes or retires the version."""
        record = self._record(name)
        entry = next(v for v in record["versions"] if v["version"] == version)
        if verified:
            entry["successes"] += 1
            entry["failures_in_a_row"] = 0
            if entry["status"] == "provisional" and entry["successes"] >= self.promote_after:
                entry["status"] = "verified"
        else:
            entry["failures"] += 1
            entry["failures_in_a_row"] += 1
            if entry["failures_in_a_row"] >= self.retire_after:
                entry["status"] = "retired"
        entry["evidence"] = (entry["evidence"]+[{"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                  "verified": bool(verified), **(evidence or {})}])[-20:]
        self._write(name, record)
        return entry["status"]

    def names(self):
        return sorted(path.parent.name for path in self.root.glob("*/skill.json")) if self.root.is_dir() else []

    def add_lessons(self, task, lessons, *, source=""):
        """Short general lessons from attempts that failed (RSIAgent's failure lessons), kept for similar tasks."""
        path = self.root/"lessons.json"
        self.root.mkdir(parents=True, exist_ok=True)
        kept = json.loads(path.read_text()) if path.is_file() else []
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for lesson in lessons:
            lesson = " ".join(str(lesson).split())[:300]
            if lesson and all(entry["lesson"] != lesson for entry in kept):
                kept.append({"lesson": lesson, "task": task[:200], "source": source, "at_utc": stamp})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(kept[-200:], indent=1)+"\n")
        tmp.replace(path)

    def lessons(self, query, k=6):
        """The kept lessons sharing the most words with the query (and its task), newest first among equals."""
        path = self.root/"lessons.json"
        if not path.is_file():
            return []
        words = lambda text: set(re.findall(r"[a-z]+", text.lower()))-{"the", "a", "an", "to", "of", "and", "it", "in", "on"}
        wanted = words(query)
        scored = [(len(wanted & words(entry["lesson"]+" "+entry["task"])), index, entry["lesson"])
                  for index, entry in enumerate(json.loads(path.read_text()))]
        return [lesson for overlap, _, lesson in sorted(scored, reverse=True)[:k] if overlap]

    def search(self, query, k=5):
        """The usable skills whose descriptions share the most words with the query, verified first."""
        words = lambda text: set(re.findall(r"[a-z]+", text.lower()))-{"the", "a", "an", "to", "of", "and", "it", "in", "on"}
        wanted = words(query)
        scored = []
        for name in self.names():
            record = self._record(name)
            usable = [v for v in record["versions"] if v["status"] != "retired"]
            if not usable:
                continue
            have = words(record["description"]+" "+name.replace("_", " "))
            overlap = len(wanted & have)
            if overlap:
                verified = any(v["status"] == "verified" for v in usable)
                scored.append((overlap/math.sqrt(len(have) or 1)+(.5 if verified else 0.), name, record["description"], verified))
        scored.sort(reverse=True)
        return [{"name": name, "description": description, "verified": verified} for _, name, description, verified in scored[:k]]
