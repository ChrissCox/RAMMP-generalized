"""Threaded call instrumentation only; actual GPU evidence is a separate run."""
import asyncio
import threading
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

from rammp_adl.motion.leasing import drain_nonpreemptible
from rammp_adl.motion.reactive_rehearsal import _ProbePlanner, _PhysicsIO
from rammp_adl.simulation import MujocoReplay


class CancellationProbeTests(unittest.IsolatedAsyncioTestCase):
    def test_quiescence_advances_physics_and_requires_measured_dwell(self):
        simulation = MujocoReplay(Path(__file__).resolve().parents[1]/'simulation/rolling_scene.xml')
        simulation.home()
        io = _PhysicsIO(simulation,SimpleNamespace(start_at=0.),None,{},asyncio.Event())
        io.supervisor_stop('TEST',0.)
        with patch('rammp_adl.motion.reactive_rehearsal.time.monotonic',return_value=.04):
            self.assertFalse(io.quiescent())
        self.assertGreater(float(simulation.data.time),.03)
        with patch('rammp_adl.motion.reactive_rehearsal.time.monotonic',return_value=.12):
            io.quiescent()
        self.assertGreater(float(simulation.data.time),.11)

    async def test_cancel_is_observed_inside_call_without_releasing_work(self):
        released = threading.Event()
        self.addCleanup(released.set)
        def step():
            if not released.wait(2.):
                raise RuntimeError('Test worker release deadline')
            return 'TEST-result'
        solver = SimpleNamespace(step=step)
        async def candidate(**kwargs):
            work = asyncio.create_task(asyncio.to_thread(solver.step))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                await drain_nonpreemptible(work)
                raise
        cancel = asyncio.Event()
        probe = _ProbePlanner(SimpleNamespace(solver=solver,candidate=candidate),cancel_during_solve=cancel)
        task = asyncio.create_task(probe.candidate())
        await asyncio.wait_for(cancel.wait(),1.)
        task.cancel()
        await asyncio.sleep(.005)
        self.assertFalse(task.done())
        self.assertNotIn('finished',probe.solver_calls[0])
        released.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        record = probe.solver_calls[0]
        self.assertLessEqual(record['started'],probe.cancel_at)
        self.assertLess(probe.cancel_at,record['finished'])
        self.assertIs(solver.step,step)

    async def test_finished_call_does_not_fabricate_inflight_cancellation(self):
        def step():
            return 'TEST-immediate'
        solver = SimpleNamespace(step=step)
        async def candidate(**kwargs):
            # Synchronous completion before the observer task can run.
            return solver.step()
        cancel = asyncio.Event()
        probe = _ProbePlanner(SimpleNamespace(solver=solver,candidate=candidate),cancel_during_solve=cancel)
        self.assertEqual(await probe.candidate(),'TEST-immediate')
        self.assertFalse(cancel.is_set())
        self.assertIsNone(probe.cancel_at)
        self.assertIs(solver.step,step)


if __name__ == '__main__':
    unittest.main()
