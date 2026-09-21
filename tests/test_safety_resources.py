import unittest

from rammp_adl.resources import ResourceError, ResourceManager
from rammp_adl.safety import ConfirmationStore, SafetyError, SafetySupervisor


class ResourceTests(unittest.TestCase):
    def test_acquisition_is_atomic_and_disjoint_work_can_overlap(self):
        resources = ResourceManager({"ARM", "PLANNER", "GRIPPER"})
        self.assertTrue(resources.try_acquire("transit", {"ARM", "PLANNER"}))
        self.assertFalse(resources.try_acquire("grasp", {"ARM", "GRIPPER"}))
        self.assertTrue(resources.try_acquire("preshape", {"GRIPPER"}))
        resources.release("transit")
        self.assertEqual(resources.owners, {"GRIPPER": "preshape"})
        with self.assertRaises(ResourceError):
            resources.assert_owned("transit", {"ARM"})

    def test_no_dynamic_model_resource(self):
        with self.assertRaises(ResourceError):
            ResourceManager({"ARM"}).try_acquire("node", {"BYPASS_SAFETY"})


class ConfirmationTests(unittest.TestCase):
    def test_assent_is_fresh_single_use_and_bound(self):
        now = [10.0]
        store = ConfirmationStore(lambda: now[0])
        kwargs = dict(task_id="task", epoch=1, node_id="step", digest="resolved-action")
        assent = store.grant(**kwargs, ttl_s=2, accepted=True)
        store.consume(assent.confirmation_id, **kwargs)
        with self.assertRaises(SafetyError):
            store.consume(assent.confirmation_id, **kwargs)
        assent = store.grant(**kwargs, ttl_s=2, accepted=True)
        now[0] = 13
        with self.assertRaises(SafetyError):
            store.consume(assent.confirmation_id, **kwargs)
        for accepted in (False, "false", "true", 1):
            with self.assertRaises(SafetyError):
                store.grant(**kwargs, ttl_s=2, accepted=accepted)


class Backend:
    mode = "simulation_fixture"

    def __init__(self, acknowledges=True):
        self.held = False
        self.acknowledges = acknowledges
        self.stops = 0

    async def stop(self):
        self.stops += 1
        self.held = self.acknowledges

    def quiescent(self):
        return self.held


class SafetyTests(unittest.IsolatedAsyncioTestCase):
    def supervisor(self, backend, clock=lambda: 0.0):
        self.epoch = 1
        def invalidate():
            self.epoch += 1
            return self.epoch
        return SafetySupervisor(backend, epoch_getter=lambda: self.epoch, invalidate_epoch=invalidate, clock=clock)

    async def test_stop_revokes_epoch_before_ack_and_is_idempotent(self):
        supervisor = self.supervisor(Backend())
        self.assertTrue(await supervisor.request_stop("user_cancel"))
        self.assertTrue(supervisor.held_verified)
        self.assertEqual(self.epoch, 2)
        await supervisor.request_stop("user_cancel")
        self.assertEqual(self.epoch, 2)
        with self.assertRaises(SafetyError):
            supervisor.check_dispatch(1)
        await supervisor.reset()
        with self.assertRaises(SafetyError):
            supervisor.check_dispatch(1)
        supervisor.check_dispatch(2)

    async def test_ack_without_measured_hold_latches_fault(self):
        supervisor = self.supervisor(Backend(acknowledges=False))
        self.assertFalse(await supervisor.request_stop("cancel"))
        self.assertTrue(supervisor.fault_latched)
        self.assertFalse(supervisor.held_verified)

    async def test_stale_heartbeat_stops_without_cloud(self):
        now = [1.0]
        supervisor = self.supervisor(Backend(), lambda: now[0])
        supervisor.heartbeat("robot_state", max_age_s=.2)
        now[0] = 2.0
        await supervisor.watch()
        self.assertTrue(supervisor.fault_latched)
        self.assertTrue(supervisor.held_verified)

    async def test_physical_mode_is_rejected_even_if_backend_claims_ready(self):
        backend = Backend()
        backend.mode = "hardware"
        supervisor = self.supervisor(backend)
        with self.assertRaises(SafetyError):
            supervisor.check_dispatch(1)


if __name__ == "__main__":
    unittest.main()
