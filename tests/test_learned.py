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


OPEN_CABINET = '''
def run(robot, part="handle", amount=1.0):
    """Open a hinged part by its handle: reach, grasp, swing it open, let go, and confirm the goal."""
    handle = robot.find(part)[0]["id"]
    robot.move_to(handle, "pregrasp")
    robot.open_hand(0.08)
    robot.move_to(handle, "grasp")
    robot.grasp(handle)
    robot.move_part(handle, amount, "rad")
    robot.release(handle)
    return robot.check("goal")
'''


@unittest.skipUnless(HAVE_JAIL, "bubblewrap and /usr/bin/python3 are needed for the jail")
class BindingTests(unittest.TestCase):
    """Learned skills against the executor: each call an admitted plan; a dry run admits the chain, moving nothing."""
    ROOT = Path(__file__).resolve().parents[1]

    def runtime(self, failures=None):
        from rammp_adl.app import fixture_runtime
        return fixture_runtime(self.ROOT/"examples/cabinet.context.json", root=self.ROOT, failures=failures)

    def host(self, runtime, kind="executor", **options):
        from rammp_adl.learned.host import DryRunHost, ExecutorHost
        return (ExecutorHost if kind == "executor" else DryRunHost)(runtime, support_of={"cabinet_handle_1": "cabinet_door_1"}, **options)

    def test_a_learned_skill_opens_the_cabinet_through_validated_steps(self):
        runtime = self.runtime()
        result = run(run_skill(OPEN_CABINET, {}, self.host(runtime)))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertTrue(result.result)
        self.assertTrue(runtime.world.goal_satisfied())
        admitted = [e for e in runtime.trace.events if e["event"] == "task_started"]
        self.assertEqual(len(admitted), 6)                                     # six motion calls, six admitted plans
        self.assertEqual(runtime.executor.resources.owners, {})

    def test_a_refused_step_reaches_the_skill_which_can_do_it_properly(self):
        source = '''
def run(robot):
    """Grasp the handle, reaching it first if the robot refuses a grasp from afar."""
    handle = robot.find("handle")[0]["id"]
    robot.open_hand(0.08)
    try:
        robot.grasp(handle)
        return "grasped at once"
    except RobotError as error:
        robot.log("refused: " + str(error)[:80])
    robot.move_to(handle, "pregrasp")
    robot.move_to(handle, "grasp")
    robot.grasp(handle)
    return robot.holding()
'''
        runtime = self.runtime()
        result = run(run_skill(source, {}, self.host(runtime)))
        self.assertEqual((result.status, result.result), ("succeeded", "cabinet_handle_1"), result.to_dict())
        self.assertTrue(result.logs[0].startswith("refused: grasp"))            # refused by admission or by the backend

    def test_a_failed_step_gets_a_fresh_epoch_and_the_skill_may_try_again(self):
        source = '''
def run(robot):
    """Open the cabinet, trying the swing twice if the first one fails."""
    handle = robot.find("handle")[0]["id"]
    robot.move_to(handle, "pregrasp")
    robot.open_hand(0.08)
    robot.move_to(handle, "grasp")
    robot.grasp(handle)
    for attempt in range(2):
        try:
            robot.move_part(handle, 1.0, "rad")
            return attempt
        except RobotError as error:
            robot.log(str(error)[:120])
'''
        runtime = self.runtime(failures={"skill_5_follow_constraint": ["model_mismatch"]})
        result = run(run_skill(source, {}, self.host(runtime)))
        self.assertEqual((result.status, result.result), ("succeeded", 1), result.to_dict())
        self.assertIn("model_mismatch", result.logs[0])
        self.assertTrue(runtime.world.goal_satisfied())

    def test_a_dry_run_moves_nothing_and_refuses_what_a_real_run_would_refuse(self):
        runtime = self.runtime()
        dry = self.host(runtime, "dry")
        result = run(run_skill(OPEN_CABINET, {}, dry))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual([step["skill"] for step in dry.steps],
                         ["move_to_pose", "set_gripper", "move_to_pose", "grasp", "follow_constraint", "release"])
        self.assertEqual(runtime.backend.events, [])                            # nothing was dispatched
        self.assertFalse(runtime.world.goal_satisfied())
        careless = '''
def run(robot):
    """Grasp the handle without reaching it."""
    robot.grasp(robot.find("handle")[0]["id"])
'''
        refused = run(run_skill(careless, {}, self.host(runtime, "dry")))
        self.assertEqual(refused.status, "failed")
        self.assertIn("would be refused", refused.error)
        for bad in ('robot.move_part("cabinet_handle_1", 3.0, "rad")', 'robot.move_to("cabinet_door_1", "grasp")', 'robot.grasp("fridge_9")'):
            source = f'def run(robot):\n    """Try something out of bounds."""\n    {bad}\n'
            self.assertEqual(run(run_skill(source, {}, self.host(runtime, "dry"))).status, "failed", bad)

    def test_the_seed_skill_opens_the_part_as_far_as_it_goes(self):
        runtime = self.runtime()
        source = (self.ROOT/"skills/learned_seeds/open_by_handle.py").read_text()
        check_source(source)
        dry = self.host(runtime, "dry")
        self.assertEqual(run(run_skill(source, {}, dry)).status, "succeeded")
        self.assertEqual(dry.steps[4]["args"]["target_value"], 1.3)                # the cabinet's hinge goes to 1.3 rad
        result = run(run_skill(source, {"amount": 1.0}, self.host(runtime)))
        self.assertEqual((result.status, result.result), ("succeeded", {"part": "cabinet_handle_1", "moved_to": 1.0, "unit": "rad"}),
                         result.to_dict())
        self.assertTrue(runtime.world.goal_satisfied())

    def test_seed_skills_compose_open_then_close_puts_the_door_back(self):
        with tempfile.TemporaryDirectory() as folder:
            library = SkillLibrary(folder)
            self.assertEqual(library.install_seeds(self.ROOT/"skills/learned_seeds"),
                             ["close_by_handle", "open_by_handle", "open_then_close"])
            self.assertEqual(library.install_seeds(self.ROOT/"skills/learned_seeds"), [])     # once
            runtime = self.runtime()
            source = library.load("open_then_close")["source"]
            dry = self.host(runtime, "dry")
            rehearsal = run(run_skill(source, {"amount": 1.0}, dry, library=library))
            self.assertEqual(rehearsal.status, "succeeded", rehearsal.to_dict())
            self.assertEqual([step["skill"] for step in dry.steps].count("follow_constraint"), 2)
            self.assertEqual([c["use"] for c in rehearsal.calls if "use" in c], ["open_by_handle", "close_by_handle"])

    def test_a_cancel_ends_the_skill_before_its_next_step(self):
        import asyncio as aio

        async def cancelled():
            cancel = aio.Event()
            cancel.set()
            return await run_skill(OPEN_CABINET, {}, self.host(self.runtime(), cancel=cancel))
        result = run(cancelled())
        self.assertEqual((result.status, result.error), ("aborted", "the task was cancelled"))


@unittest.skipUnless(HAVE_JAIL, "bubblewrap and /usr/bin/python3 are needed for the jail")
class LoopTests(unittest.TestCase):
    """Voyager's loop: every refusal and failure goes back to the writer; only a verified skill is kept."""
    ROOT = Path(__file__).resolve().parents[1]

    def test_the_writer_learns_from_the_gate_the_rehearsal_and_keeps_the_verified_skill(self):
        from rammp_adl.app import fixture_runtime
        from rammp_adl.learned.host import DryRunHost, ExecutorHost
        from rammp_adl.learned.loop import learn_skill
        answers = [
            "```python\n# name: open_door\nimport os\ndef run(robot):\n    \"\"\"Open.\"\"\"\n    os.system('x')\n```",
            "```python\n# name: open_door\ndef run(robot):\n    \"\"\"Open.\"\"\"\n    robot.grasp('cabinet_handle_1')\n```",
            "Here it is.\n```python\n# name: open_hinged_part\n"+OPEN_CABINET.strip()+"\n```"]

        class Writer:
            prompts = []

            async def write(self, text):
                Writer.prompts.append(text)
                return answers[min(len(Writer.prompts), len(answers))-1]
        runtime = fixture_runtime(self.ROOT/"examples/cabinet.context.json", root=self.ROOT)
        support = {"cabinet_handle_1": "cabinet_door_1"}

        async def rehearse(source):
            return await run_skill(source, {}, DryRunHost(runtime, support_of=support))

        async def execute(source):
            return await run_skill(source, {}, ExecutorHost(runtime, support_of=support))

        async def verify(run):
            met = runtime.world.goal_satisfied()
            return met, "the goal is measured met" if met else "the goal is not met"
        with tempfile.TemporaryDirectory() as folder:
            library = SkillLibrary(folder)
            story = run(learn_skill("open the cabinet door", writer=Writer(), library=library, scene=[{"id": "cabinet_handle_1"}],
                                    rehearse=rehearse, execute=execute, verify=verify))
            self.assertEqual(story["status"], "learned", story)
            self.assertEqual([r.get("stage") for r in story["rounds"]], ["gate", "rehearsal", "kept"])
            self.assertIn("import os is not allowed", Writer.prompts[1])           # the gate's refusal went back
            self.assertIn("In rehearsal (nothing moved)", Writer.prompts[2])       # so did the refused grasp
            self.assertIn("robot.grasp", Writer.prompts[2])
            self.assertEqual(library.load("open_hinged_part")["version"], 1)
            self.assertTrue(runtime.world.goal_satisfied())
            again = run(learn_skill("open the cabinet door", writer=Writer(), library=library, scene=[], rehearse=rehearse))
            self.assertIn("open_hinged_part", Writer.prompts[-1])                  # the kept skill is offered next time

    def test_candidates_are_rehearsed_side_by_side_one_runs_and_failures_become_lessons(self):
        from rammp_adl.app import fixture_runtime
        from rammp_adl.learned.host import DryRunHost, ExecutorHost
        from rammp_adl.learned.loop import learn_skill
        careless = "```python\n# name: grab_first\ndef run(robot):\n    \"\"\"Grab the handle at once.\"\"\"\n    robot.grasp('cabinet_handle_1')\n```"
        slow = ("```python\n# name: open_slowly\ndef run(robot):\n    \"\"\"Open the door in two pulls.\"\"\"\n"
                "    h = 'cabinet_handle_1'\n    robot.move_to(h, 'pregrasp')\n    robot.open_hand(0.08)\n    robot.move_to(h, 'grasp')\n"
                "    robot.grasp(h)\n    robot.move_part(h, 0.5, 'rad')\n    robot.move_part(h, 1.0, 'rad')\n    robot.release(h)\n```")
        good = "```python\n# name: open_hinged_part\n"+OPEN_CABINET.strip()+"\n```"
        answers = iter(["Three ways:\n"+careless+"\n"+good+"\n"+slow, "- Reach the grasp pose before grasping; a grasp from afar is refused."])

        class Writer:
            prompts = []

            async def write(self, text):
                Writer.prompts.append(text)
                return next(answers)
        runtime = fixture_runtime(self.ROOT/"examples/cabinet.context.json", root=self.ROOT)
        support = {"cabinet_handle_1": "cabinet_door_1"}
        rehearsed, executed, offered = [], [], []

        async def rehearse(source):
            rehearsed.append(source)
            return await run_skill(source, {}, DryRunHost(runtime, support_of=support))

        async def execute(source):
            executed.append(source)
            return await run_skill(source, {}, ExecutorHost(runtime, support_of=support))

        async def verify(run):
            return runtime.world.goal_satisfied(), "goal"

        async def choose(task, candidates):
            offered.append([name for name, _ in candidates])
            return [name for name, _ in candidates].index("open_slowly")        # Jev prefers the two-pull opening
        with tempfile.TemporaryDirectory() as folder:
            library = SkillLibrary(folder)
            story = run(learn_skill("open the cabinet door", writer=Writer(), library=library, scene=[], rehearse=rehearse,
                                    execute=execute, verify=verify, choose=choose))
            self.assertEqual((story["status"], story["name"]), ("learned", "open_slowly"), story)
            self.assertEqual(len(rehearsed), 3)                                    # all three rehearsed, one round
            self.assertEqual(offered, [["open_hinged_part", "open_slowly"]])       # the careless one failed rehearsal
            self.assertEqual(len(executed), 1)
            self.assertIn("up to 3 different approaches", Writer.prompts[0])
            self.assertEqual(story["lessons"], ["Reach the grasp pose before grasping; a grasp from afar is refused."])
            self.assertEqual(library.lessons("open a drawer by grasping its handle"), story["lessons"])
            self.assertIn("Lessons from earlier attempts", __import__("rammp_adl.learned.loop", fromlist=["prompt"]).prompt(
                "grasp the mug", scene=[], examples=[], feedback="", lessons=library.lessons("grasp the mug")))

    def test_the_curriculum_proposes_practice_from_the_scene_and_what_the_library_lacks(self):
        from rammp_adl.learned.loop import WriterError, propose_tasks

        class Writer:
            asked = []

            async def write(self, text):
                Writer.asked.append(text)
                return ('Here: [{"task": "open the drawer below the cabinet", "why": "no slide skill yet", "undo": "close the drawer"},'
                        ' {"task": "press the microwave button", "why": "no press skill", "undo": "none needed"},'
                        ' {"task": "x", "why": "", "undo": ""}, {"why": "no task"}]')
        with tempfile.TemporaryDirectory() as folder:
            library = SkillLibrary(folder)
            library.save("pick_up", PICK)
            library.add_lessons("pick up the cup", ["Reach the grasp pose before grasping."])
            proposals = run(propose_tasks(Writer(), library, [{"id": "cup_1", "label": "cup"}], count=2))
        self.assertEqual([p["task"] for p in proposals], ["open the drawer below the cabinet", "press the microwave button"])
        self.assertIn("pick_up (provisional)", Writer.asked[0])                 # it knows what the robot can do
        self.assertIn("Reach the grasp pose", Writer.asked[0])

        class Silent:
            async def write(self, text):
                return "I would practise opening things."
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(WriterError):
            run(propose_tasks(Silent(), SkillLibrary(folder), []))

    def test_the_writers_answer_is_read_from_its_python_block(self):
        from rammp_adl.learned.loop import WriterError, extract_skill
        self.assertEqual(extract_skill("x\n```python\n# name: pick_up\ndef run(robot):\n    pass\n```"),
                         ("pick_up", "# name: pick_up\ndef run(robot):\n    pass\n"))
        with self.assertRaises(WriterError):
            extract_skill("no code here")


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
