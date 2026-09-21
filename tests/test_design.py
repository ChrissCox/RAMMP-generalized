import copy
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_design as c


class ContractRejections(unittest.TestCase):
    def setUp(self):
        self.library = c.read(c.ROOT / "skills/adl_skill_library.yaml")
        self.plan = c.read(c.ROOT / "examples/cabinet.plan.json")
        self.context = c.read(c.ROOT / "examples/cabinet.context.json")

    def rejects(self, change):
        change(self.plan)
        with self.assertRaises(ValueError):
            c.check_plan(self.plan, self.library, self.context)

    def test_valid_examples(self):
        for path in (c.ROOT / "examples").glob("*.plan.json"):
            context = c.read(path.with_name(path.name.replace(".plan.json", ".context.json")))
            self.assertTrue(c.check_plan(c.read(path), self.library, context))

    def test_duplicate_json_key(self):
        with self.assertRaises(ValueError):
            c.loads('{"x": 1, "x": 2}')

    def test_nonfinite_json(self):
        for text in ('NaN', 'Infinity', '-Infinity'):
            with self.assertRaises(ValueError):
                c.loads(text)

    def test_model_cannot_override_claims(self):
        self.rejects(lambda p: p["nodes"][0].update(claims=[]))

    def test_model_cannot_override_success(self):
        self.rejects(lambda p: p.update(success=True))

    def test_unknown_skill(self):
        self.rejects(lambda p: p["nodes"][0].update(skill="run_python"))

    def test_unavailable_skill(self):
        self.context["available_skills"].remove("grasp")
        with self.assertRaises(ValueError):
            c.check_plan(self.plan, self.library, self.context)

    def test_stale_epoch(self):
        self.rejects(lambda p: p.update(execution_epoch=p["execution_epoch"] + 1))

    def test_wrong_hash(self):
        self.rejects(lambda p: p.update(skill_library_hash="wrong"))

    def test_unknown_entity(self):
        self.rejects(lambda p: p["nodes"][0]["args"].update(entity_id="other-cabinet"))

    def test_unknown_arg(self):
        self.rejects(lambda p: p["nodes"][0]["args"].update(python="execute()"))

    def test_duplicate_node(self):
        self.rejects(lambda p: p["nodes"].append(copy.deepcopy(p["nodes"][0])))

    def test_cycle(self):
        self.rejects(lambda p: p["edges"].append({"from": p["nodes"][-1]["id"],
                                                 "to": p["nodes"][0]["id"]}))

    def test_unknown_edge(self):
        self.rejects(lambda p: p["edges"].append({"from": "missing", "to": p["nodes"][0]["id"]}))

    def test_invalid_nested_pose(self):
        def change(p):
            node = next(n for n in p["nodes"] if n["skill"] == "move_to_pose")
            node["args"]["target"] = "an opaque pose string"
        self.rejects(change)

    def test_boolean_is_not_number(self):
        def change(p):
            node = next(n for n in p["nodes"] if n["skill"] == "set_gripper")
            node["args"]["aperture_m"] = True
        self.rejects(change)

    def test_constraint_units(self):
        def change(p):
            node = next(n for n in p["nodes"] if n["skill"] == "follow_constraint")
            node["args"]["target_unit"] = "m"
        self.rejects(change)

    def test_constraint_bounds(self):
        def change(p):
            node = next(n for n in p["nodes"] if n["skill"] == "follow_constraint")
            node["args"]["target_value"] = 9.0
        self.rejects(change)

    def test_invalid_constraint_evidence(self):
        self.context["constraints"][0]["validity"] = "unknown"
        with self.assertRaises(ValueError):
            c.check_plan(self.plan, self.library, self.context)

    def test_wrong_goal_unit(self):
        self.context["goal"]["args"]["target_unit"] = "m"
        with self.assertRaises(ValueError):
            c.check_plan(self.plan, self.library, self.context)

    def test_epoch_exceeds_transport_range(self):
        self.context["execution_epoch"] = 2**64
        self.rejects(lambda p: p.update(execution_epoch=2**64))

    def test_unbound_predicate_argument(self):
        self.library["skills"][0]["preconditions"][0]["bindings"]["entity_id"] = {
            "arg_path": ["invented"]}
        with self.assertRaises(ValueError):
            c.check_catalog(self.library)

    def test_no_plan_response(self):
        response = {"result": {"status": "NEED_CAPABILITY", "detail": "No supported contact handler"}}
        c.validate(response, c.response_schema(self.library))
        response["result"]["plan"] = self.plan
        with self.assertRaises(ValueError):
            c.validate(response, c.response_schema(self.library))

    def test_unknown_resource_in_catalog(self):
        self.library["skills"][0]["claims"].append("MISSING")
        with self.assertRaises(ValueError):
            c.check_catalog(self.library)

    def test_duplicate_skill_in_catalog(self):
        self.library["skills"].append(copy.deepcopy(self.library["skills"][0]))
        with self.assertRaises(ValueError):
            c.check_catalog(self.library)


if __name__ == "__main__":
    unittest.main()
