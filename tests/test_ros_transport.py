"""Pure clock/thread bridge tests; these are not ROS compilation/transport tests."""
import asyncio
import threading
import unittest

from rammp_adl.contracts import ContractError
from rammp_adl.ros_node import AsyncWorker, MonotonicRosClockMapping


class AsyncWorkerTests(unittest.TestCase):
    def test_coroutine_uses_dedicated_running_event_loop(self):
        worker = AsyncWorker()
        try:
            async def identify():
                return threading.current_thread().name, asyncio.get_running_loop()
            name, loop = worker.call(identify(), timeout=1.)
            self.assertEqual(name, "rammp-runtime")
            self.assertIs(loop, worker.loop)
        finally:
            self.assertTrue(worker.close())
        self.assertFalse(worker.thread.is_alive())

    def test_timeout_cancels_pending_work_without_claiming_result(self):
        worker = AsyncWorker()
        cancelled = threading.Event()
        async def pending():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        try:
            with self.assertRaisesRegex(RuntimeError, "bounded deadline"):
                worker.call(pending(), timeout=.02)
            self.assertTrue(cancelled.wait(timeout=1.))
        finally:
            self.assertTrue(worker.close())

    def test_shutdown_drains_coroutine_and_rejects_late_submission(self):
        worker = AsyncWorker()
        entered, drained = threading.Event(), threading.Event()
        async def pending():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        future = worker.submit(pending())
        self.assertTrue(entered.wait(timeout=1.))
        self.assertTrue(worker.close())
        self.assertTrue(drained.is_set())
        self.assertTrue(future.done())
        async def late():
            return 1
        with self.assertRaisesRegex(RuntimeError, "closing"):
            worker.submit(late())
        self.assertTrue(worker.close())


class RosClockMappingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = MonotonicRosClockMapping(100., 1_000_000_000_000)

    def test_original_capture_age_is_preserved(self):
        capture, expiry = self.mapping.capture_and_expiry(100.1, 1., monotonic_now=100.4, ros_now_ns=1_000_400_000_000)
        self.assertEqual(capture, 1_000_100_000_000)
        self.assertEqual(expiry, 1_001_100_000_000)
        self.assertNotEqual(capture, 1_000_400_000_000)

    def test_expired_and_future_evidence_rejected(self):
        for capture in (99., 100.5):
            with self.assertRaises(ContractError):
                self.mapping.capture_and_expiry(capture, 1., monotonic_now=100.4, ros_now_ns=1_000_400_000_000)

    def test_ros_clock_jump_and_pause_reject_mapping(self):
        for ros_now in (1_002_000_000_000, 1_000_000_000_000):
            with self.assertRaisesRegex(ContractError, "clock mapping"):
                self.mapping.capture_and_expiry(100.2, 1., monotonic_now=100.5, ros_now_ns=ros_now)


if __name__ == "__main__":
    unittest.main()
