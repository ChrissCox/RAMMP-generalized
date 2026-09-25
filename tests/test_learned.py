"""Learned skills: the gate, the jail's isolation, the call protocol and budget, composition, and the library."""
import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rammp_adl.learned import GateError, SkillLibrary, check_source, run_skill, validate_call
from rammp_adl.learned.api import PrimitiveError
from rammp_adl.learned.jail import SkillAbort, jail_command

HAVE_JAIL = shutil.which("bwrap") is not None and Path("/usr/bin/python3").exists()

PICK = '''
import math

def run(robot, thing="cup"):
    """Pick up a thing: find it, reach it, grasp it, and say where it was."""
    found = robot.find(thing)
    if not found:
        raise RobotError("no " + thing + " in view")
    target = found[0]
    robot.open_hand(0.08)
    robot.move_to(target["id"], "pregrasp")
    robot.move_to(target["id"], role="grasp")
    robot.grasp(target["id"])
    print("holding", robot.holding())
    return {"picked": target["id"], "distance": round(math.dist(target["position_m"], [0, 0, 0]), 3)}
'''


class Host:
    """A binding that records calls and answers from a script; the real one runs validated plans."""
    def __init__(self, answers=None, fail=None, abort=None):
        self.calls, self.answers, self.fail, self.abort = [], answers or {}, fail or {}, abort

    async def call(self, name, args):
        self.calls.append((name, args))
        if name == self.abort:
            raise SkillAbort("the operator cancelled")
        if name in self.fail:
            raise RuntimeError(self.fail[name])
        return self.answers.get(name)


def run(coroutine):
    return asyncio.run(coroutine)


class GateTests(unittest.TestCase):
    def test_a_plain_skill_passes_and_its_docstring_is_its_description(self):
        self.assertEqual(check_source(PICK), "Pick up a thing: find it, reach it, grasp it, and say where it was.")

    def test_the_ways_out_of_an_interpreter_are_refused(self):
        body = 'def run(robot):\n    """x"""\n    {}\n'
        for line in ("import os", "import subprocess", "from socket import socket", "().__class__.__bases__",
                     "open('/home/abra/.profile')", "eval('1')", "exec('1')", "getattr(robot, 'x')", "__import__('os')",
                     "globals()", "x = '{0.__class__}'.format(robot)", "type(robot)", "compile('1', 'x', 'exec')"):
            with self.subTest(line=line), self.assertRaises(GateError):
                check_source(body.format(line))
        for source in ("def act(robot):\n    pass\n", "def run(robot):\n    pass\n", "def run(arm):\n    '''x'''\n",
                       "class Thing:\n    pass\ndef run(robot):\n    '''x'''\n", "print(1)\ndef run(robot):\n    '''x'''\n",
                       "def run(robot):\n    '''x'''\n    global y\n"):
            with self.subTest(source=source), self.assertRaises(GateError):
                check_source(source)


class ApiTests(unittest.TestCase):
    def test_arguments_come_from_the_table_never_from_the_skill(self):
        self.assertEqual(validate_call("move_to", {"object": "cup_1", "role": "grasp"}), {"object": "cup_1", "role": "grasp"})
        for name, args in (("move_to", {"object": "cup_1", "role": "grasp", "speed": 3.}), ("open_hand", {"aperture_m": .2}),
                           ("move_to", {"object": "cup_1", "role": "inside"}), ("fly", {}), ("move_part", {"object": "d", "amount": 1.}),
                           ("open_hand", {"aperture_m": True}), ("move_part", {"object": "d", "amount": float("nan"), "unit": "rad"})):
            with self.subTest(name=name, args=args), self.assertRaises(PrimitiveError):
                validate_call(name, args)


@unittest.skipUnless(HAVE_JAIL, "bubblewrap and /usr/bin/python3 are needed for the jail")
class JailTests(unittest.TestCase):
    def test_a_skill_runs_its_calls_through_the_host_in_order_and_returns(self):
        host = Host({"find": [{"id": "cup_1", "label": "cup", "position_m": [.3, .4, 0.]}], "holding": "cup_1"})
        result = run(run_skill(PICK, {"thing": "cup"}, host, name="pick_up"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(result.result, {"picked": "cup_1", "distance": .5})
        self.assertEqual([c[0] for c in host.calls], ["find", "open_hand", "move_to", "move_to", "grasp", "holding"])
        self.assertEqual(host.calls[3], ("move_to", {"object": "cup_1", "role": "grasp"}))
        self.assertEqual(result.logs, ["holding cup_1"])

    def test_a_refused_or_failed_call_reaches_the_skill_which_may_handle_it(self):
        source = '''
def run(robot):
    """Try a bad role, then a failing grasp, and report both."""
    notes = []
    try:
        robot.move_to("cup_1", "inside")
    except RobotError as error:
        notes.append(str(error))
    try:
        robot.grasp("cup_1")
    except RobotError as error:
        notes.append(str(error))
    return notes
'''
        host = Host(fail={"grasp": "the gripper closed on nothing"})
        result = run(run_skill(source, {}, host))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertIn("expected one of", result.result[0])
        self.assertIn("closed on nothing", result.result[1])
        self.assertEqual([c[0] for c in host.calls], ["grasp"])                # the bad role never reached the host
        self.assertIn("refused", result.calls[0])

    def test_the_budget_the_clock_the_cpu_and_an_abort_end_a_run(self):
        spinner = 'def run(robot):\n    """Ask forever."""\n    while True:\n        robot.holding()\n'
        self.assertEqual(run(run_skill(spinner, {}, Host(), max_calls=5)).status, "budget")
        busy = 'def run(robot):\n    """Think forever."""\n    while True:\n        pass\n'
        self.assertEqual(run(run_skill(busy, {}, Host(), timeout_s=2.)).status, "timeout")
        self.assertEqual(run(run_skill(busy, {}, Host(), timeout_s=20., cpu_s=1)).status, "crashed")
        hog = 'def run(robot):\n    """Eat memory."""\n    x = [0] * (10 ** 10)\n'
        self.assertEqual(run(run_skill(hog, {}, Host(), memory_mb=256)).status, "failed")
        stopped = run(run_skill(PICK, {}, Host({"find": [{"id": "c", "position_m": [0, 0, 0]}]}, abort="move_to")))
        self.assertEqual((stopped.status, stopped.error), ("aborted", "the operator cancelled"))

    def test_the_jail_has_no_network_no_host_files_no_ros_no_environment_and_no_writes_outside_tmp(self):
        # The operating-system layer on its own: an unrestricted script in the same jail the skills run in.
        probe = '''
import json, os, socket, subprocess
report = {}
try:
    socket.create_connection(("1.1.1.1", 80), timeout=2); report["network"] = "open"
except OSError: report["network"] = "blocked"
report["profile"] = os.path.exists("/home/abra/.profile")
report["ros"] = os.path.exists("/opt/ros")
report["repo"] = os.path.exists("/home/abra/RAMMP-generalized")
report["env"] = sorted(k for k in os.environ if k not in ("PWD", "LC_CTYPE"))
try:
    open("/usr/evil", "w"); report["usr_write"] = "open"
except OSError: report["usr_write"] = "blocked"
try:
    subprocess.run(["ps", "-e"], capture_output=True, text=True, timeout=5); report["pids"] = len(os.listdir("/proc"))
except Exception as exc: report["pids"] = str(exc)
print(json.dumps(report))
'''
        command = jail_command()[:-1]+[probe]
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, env={})
        report = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual((report["network"], report["profile"], report["ros"], report["repo"], report["env"], report["usr_write"]),
                         ("blocked", False, False, False, [], "blocked"))

    def test_skills_use_other_skills_from_the_library(self):
        with tempfile.TemporaryDirectory() as folder:
            library = SkillLibrary(folder)
            library.save("pick_up", PICK)
            tidy = ('def run(robot, things=("cup",)):\n    """Pick up each thing in turn."""\n'
                    '    return [robot.use("pick_up", thing=thing)["picked"] for thing in things]\n')
            host = Host({"find": [{"id": "cup_1", "position_m": [0., 0., 0.]}], "holding": "cup_1"})
            result = run(run_skill(tidy, {"things": ["cup", "mug"]}, host, library=library))
            self.assertEqual(result.status, "succeeded", result.to_dict())
            self.assertEqual(result.result, ["cup_1", "cup_1"])
            self.assertEqual([c.get("use") for c in result.calls if "use" in c], ["pick_up", "pick_up"])
            missing = 'def run(robot):\n    """Use what is not there."""\n    return robot.use("fly")\n'
            self.assertIn("cannot use fly", run(run_skill(missing, {}, host, library=library)).error)


class LibraryTests(unittest.TestCase):
    def test_versions_are_kept_promoted_by_verified_runs_retired_by_failures_and_found_by_what_they_do(self):
        with tempfile.TemporaryDirectory() as folder:
            library = SkillLibrary(folder, promote_after=2, retire_after=2)
            self.assertEqual(library.save("pick_up", PICK, task="pick up the cup"), 1)
            self.assertEqual(library.save("pick_up", PICK.replace("0.08", "0.07")), 2)
            with self.assertRaises(GateError):
                library.save("bad", "import os\ndef run(robot):\n    '''x'''\n")
            self.assertEqual(library.load("pick_up")["version"], 2)
            for _ in range(2):
                status = library.record_outcome("pick_up", 1, verified=True, evidence={"task": "pick up the cup"})
            self.assertEqual(status, "verified")
            self.assertEqual(library.load("pick_up")["version"], 1)                # verified beats newer provisional
            for _ in range(2):
                status = library.record_outcome("pick_up", 1, verified=False)
            self.assertEqual(status, "retired")
            self.assertEqual(library.load("pick_up")["version"], 2)
            self.assertEqual([hit["name"] for hit in library.search("grasp the mug")], ["pick_up"])
            self.assertEqual(library.search("open the drawer"), [])


if __name__ == "__main__":
    unittest.main()
