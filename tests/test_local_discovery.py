"""Local discovery: the detector's boxes as Astra-shaped entities, and Astra only when the target is not seen."""
import asyncio
import unittest
from types import SimpleNamespace

from rammp_adl.perception.local_discovery import LocalFirstReasoner, discover, task_phrases


class Detector:
    """Answers each query with scripted boxes: {query: [(score, box)]}."""
    def __init__(self, found):
        self.found, self.asked = found, []

    def detect(self, image, queries, *, threshold=.1):
        self.asked.append(list(queries))
        return [(queries.index(query), score, box) for query, hits in self.found.items() if query in queries
                for score, box in hits if score >= threshold]


DOOR = {"door handle": [(.76, [.64, .55, .67, .83])], "knob": [(.33, [.30, .40, .33, .45])],
        "cabinet door": [(.48, [.27, .01, .69, .99]), (.27, [.19, .0, .75, .99]), (.26, [.74, .0, .96, .98])],
        "cabinet": [(.5, [.1, .0, .9, 1.])]}


class DiscoveryTests(unittest.TestCase):
    def test_the_task_words_and_the_vocabulary_are_asked_in_one_pass(self):
        self.assertEqual(task_phrases("open the cabinet door in front of you"), ["cabinet door", "cabinet", "door"])
        detector = Detector(DOOR)
        discover(detector, None, "k1", "open the cabinet door")
        self.assertEqual(len(detector.asked), 1)
        self.assertIn("cup", detector.asked[0])
        self.assertIn("cabinet door", detector.asked[0])

    def test_a_handle_is_attached_to_the_door_holding_it_and_weak_duplicates_are_dropped(self):
        entities, visible = discover(Detector(DOOR), None, "k1", "open the cabinet door in front of you")
        self.assertTrue(visible)
        labels = [(e["label"], e["kind"], e["attached_to"]) for e in entities]
        self.assertEqual(labels[0], ("door handle", "handle", "cabinet door"))
        self.assertNotIn(("knob", "handle", "cabinet door"), labels)            # under half the best handle's score
        self.assertEqual(sum(1 for label in labels if label[0] == "cabinet door"), 1)   # the others are inside it or weak
        self.assertNotIn("free_object", [e["kind"] for e in entities])          # "cabinet" is not a thing to pick up here
        handle = entities[0]
        self.assertEqual(handle["grasp_point_xy_normalized"], [.655, .69])
        self.assertTrue(handle["grasp_point_given"])

    def test_the_target_must_be_seen_and_an_unknown_thing_to_pick_up_is_a_free_object(self):
        _, visible = discover(Detector(DOOR), None, "k1", "pick up the cup")
        self.assertFalse(visible)
        entities, visible = discover(Detector({"teddy bear": [(.2, [.4, .4, .6, .7])]}), None, "k1", "pick up the teddy bear")
        self.assertTrue(visible)
        self.assertEqual((entities[0]["label"], entities[0]["kind"]), ("teddy bear", "free_object"))


class Astra:
    def __init__(self):
        self.calls = []

    async def discover_scene(self, context, images, *, task_text="", **kwargs):
        self.calls.append(("discover", task_text))
        return SimpleNamespace(status="OK", candidates=(), proposal={"target_visible": False, "search_hint": "left", "search_note": ""})

    async def ground_target(self, context, entity_id, images, *, query="", **kwargs):
        self.calls.append(("ground", entity_id))
        return SimpleNamespace(status="NO_DETECTION", candidates=())

    def close(self):
        return "closed"


class LocalFirstTests(unittest.TestCase):
    def reasoner(self, found):
        astra = Astra()
        local = LocalFirstReasoner(astra, Detector(found))
        local._image = staticmethod(lambda crop: None)
        return astra, local

    def test_what_the_detector_sees_needs_no_request_and_what_it_does_not_goes_to_astra(self):
        crop = SimpleNamespace(image_id="k1", jpeg_bytes=b"")
        astra, local = self.reasoner(DOOR)
        result = asyncio.run(local.discover_scene({}, [crop], task_text="open the cabinet door"))
        self.assertEqual((result.status, result.model_id, astra.calls), ("OK", "owlv2", []))
        self.assertEqual(result.candidates[0]["label"], "door handle")
        result = asyncio.run(local.discover_scene({}, [crop], task_text="pick up the cup"))
        self.assertEqual(astra.calls, [("discover", "pick up the cup")])       # with Astra's search hint
        self.assertEqual(result.proposal["search_hint"], "left")
        self.assertEqual(local.close(), "closed")                             # everything else is Astra's

    def test_grounding_finds_the_entity_by_its_label_else_asks_astra(self):
        crop = SimpleNamespace(image_id="k2", jpeg_bytes=b"")
        context = {"entities": [{"entity_id": "door_handle_1", "label": "door handle"}, {"entity_id": "cup_1", "label": "cup"}]}
        astra, local = self.reasoner(DOOR)
        result = asyncio.run(local.ground_target(context, "door_handle_1", [crop]))
        self.assertEqual((result.status, result.candidates[0]["entity_id"], astra.calls), ("OK", "door_handle_1", []))
        asyncio.run(local.ground_target(context, "cup_1", [crop]))
        self.assertEqual(astra.calls, [("ground", "cup_1")])


if __name__ == "__main__":
    unittest.main()
