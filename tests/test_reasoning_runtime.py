"""Provider-independent fault injection at the held Astra gateway."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from rammp_adl.contracts import Catalog, ContractError, strict_loads, validate_schema
from rammp_adl.reasoning import AstraReasoner, OpenAIResponsesTransport, _provider_schema

ROOT = Path(__file__).resolve().parents[1]

def response(data=None, *, raw=None, status='completed'):
    text = raw if raw is not None else json.dumps(data)
    return {'status': status, 'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                                        'content': [{'type': 'output_text', 'text': text}]}]}

class ScriptedTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
    async def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return await item()
        return copy.deepcopy(item)

class HTTPFailure(Exception):
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.response = SimpleNamespace(headers={} if retry_after is None else {'retry-after': retry_after})
        super().__init__('private provider detail must not leak')

class TestCrop:
    image_id = 'image-1'
    def __init__(self, *, face=False, size=100, edge=64):
        self.face, self.size, self.edge = face, size, edge
    def validate_for_egress(self, *, max_bytes, max_long_edge, allow_face):
        if self.size > max_bytes or self.edge > max_long_edge or (self.face and not allow_face):
            raise ValueError('image policy')
    def cloud_metadata(self):
        return {'image_id': self.image_id, 'width': 64, 'height': 64}
    def as_openai_input(self):
        return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,/9j/2Q==', 'detail': 'low'}

class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.context = strict_loads((ROOT / 'examples/cabinet.context.json').read_bytes())
        self.plan = strict_loads((ROOT / 'examples/cabinet.plan.json').read_bytes())
        self.epoch = self.context['execution_epoch']
        self.held = True
    def gateway(self, *outcomes, config=None):
        self.transport = ScriptedTransport(*outcomes)
        return AstraReasoner(self.catalog, config, transport=self.transport,
                             hold_assertion=lambda: self.held, epoch_getter=lambda task: self.epoch)
    def ok(self):
        return response({'result': {'status': 'OK', 'plan': self.plan}})
    async def test_strict_pinned_request_and_no_robot_tools(self):
        gateway = self.gateway(self.ok())
        result = await gateway.generate_plan(self.context, task_text='open the cabinet', request_id='request-1')
        self.assertEqual(result.status, 'OK', result.detail)
        self.assertEqual(result.plan, self.plan)
        self.assertEqual(result.request_id, 'request-1')
        request = self.transport.requests[0]
        self.assertEqual(request['model'], 'gpt-6-astra')
        self.assertIs(request['store'], False)
        self.assertNotIn('tools', request)
        self.assertTrue(request['text']['format']['strict'])
        schema_text = json.dumps(request['text']['format']['schema'])
        self.assertIn(self.context['task_id'], schema_text)
        self.assertIn(self.context['snapshot_id'], schema_text)
        self.assertIn(self.catalog.hash, schema_text)
        self.assertEqual(result.provider_requests, 1)
        def explicit_scalars(schema):
            if 'const' in schema or 'enum' in schema:
                self.assertIn('type', schema)
            for child in schema.get('properties', {}).values():
                explicit_scalars(child)
            for child in schema.get('anyOf', []):
                explicit_scalars(child)
            if 'items' in schema:
                explicit_scalars(schema['items'])
        explicit_scalars(request['text']['format']['schema'])
    async def test_unheld_never_calls_provider(self):
        gateway = self.gateway(self.ok())
        self.held = False
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'UNAVAILABLE')
        self.assertEqual(self.transport.requests, [])
    async def test_stale_epoch_never_calls_provider(self):
        gateway = self.gateway(self.ok())
        self.epoch += 1
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'UNAVAILABLE')
        self.assertEqual(self.transport.requests, [])
    async def test_epoch_change_discards_late_response(self):
        async def change():
            self.epoch += 1
            return self.ok()
        gateway = self.gateway(change)
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertIsNone(result.plan)
    async def test_hold_loss_during_provider_wait(self):
        cancelled = asyncio.Event()
        async def lose_hold():
            self.held = False
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
        gateway = self.gateway(lose_hold)
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'UNAVAILABLE')
        await asyncio.wait_for(cancelled.wait(), 0.5)
    async def test_refusal_does_not_regenerate(self):
        refusal = {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'no'}]}]}
        gateway = self.gateway(refusal)
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'REFUSED')
        self.assertIsNone(result.plan)
        self.assertEqual(len(self.transport.requests), 1)
    async def test_typed_no_plan(self):
        for status in ['NEED_CAPABILITY', 'INFEASIBLE', 'NEED_OBSERVATION', 'AMBIGUOUS', 'REFUSED']:
            gateway = self.gateway(response({'result': {'status': status, 'detail': 'explanation'}}))
            result = await gateway.generate_plan(self.context)
            self.assertEqual(result.status, status)
            self.assertIsNone(result.plan)
    async def test_duplicate_keys_regenerate_once(self):
        gateway = self.gateway(response(raw='{"result":{},"result":{}}'), self.ok())
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'OK')
        self.assertEqual(result.provider_requests, 2)
        self.assertIn('Prior response failed', self.transport.requests[1]['input'][0]['content'][0]['text'])
    async def test_incomplete_and_nonfinite_are_never_plans(self):
        for bad in [response(self.plan, status='incomplete'), response(raw='{"result":NaN}'), response(raw='not json')]:
            gateway = self.gateway(bad, bad)
            result = await gateway.generate_plan(self.context)
            self.assertEqual(result.status, 'INVALID_OUTPUT')
            self.assertIsNone(result.plan)
            self.assertEqual(result.provider_requests, 2)
    async def test_wrong_plan_epoch_rejected(self):
        wrong = copy.deepcopy(self.plan)
        wrong['execution_epoch'] += 1
        gateway = self.gateway(response({'result': {'status': 'OK', 'plan': wrong}}), config={'max_plan_regenerations': 0})
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'INVALID_OUTPUT')
    async def test_unavailable_skill_not_in_output_schema(self):
        self.context['available_skills'] = ['observe']
        gateway = self.gateway(self.ok(), config={'max_plan_regenerations': 0})
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'INVALID_OUTPUT')
        self.assertNotIn('"move_to_pose"', json.dumps(self.transport.requests[0]['text']['format']['schema']))
    async def test_no_capabilities_short_circuit(self):
        self.context['available_skills'] = []
        gateway = self.gateway()
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'NEED_CAPABILITY')
        self.assertEqual(self.transport.requests, [])
    async def test_transport_retry_once(self):
        gateway = self.gateway(ConnectionError('private detail'), self.ok())
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'OK')
        self.assertEqual(result.provider_requests, 2)
    async def test_auth_and_model_unavailable_never_retry_or_fallback(self):
        for code in [401, 403, 404, 400]:
            gateway = self.gateway(HTTPFailure(code))
            result = await gateway.generate_plan(self.context)
            self.assertEqual(result.status, 'UNAVAILABLE')
            self.assertNotIn('private', result.detail)
            self.assertEqual(len(self.transport.requests), 1)
    async def test_retry_after_must_fit_event_deadline(self):
        for delay in ('999', 'NaN', 'Infinity'):
            gateway = self.gateway(HTTPFailure(429, delay))
            result = await gateway.generate_plan(self.context)
            self.assertEqual(result.status, 'RATE_LIMITED')
            self.assertEqual(len(self.transport.requests), 1)
    async def test_short_retry_after_can_succeed(self):
        gateway = self.gateway(HTTPFailure(429, '0'), self.ok())
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'OK')
        self.assertEqual(len(self.transport.requests), 2)
    async def test_timeout_and_external_cancellation_are_bounded(self):
        async def hang():
            await asyncio.sleep(10)
            return self.ok()
        gateway = self.gateway(hang, config={'request_timeout_s': 0.02, 'event_deadline_s': 0.2, 'max_transport_retries': 0})
        result = await asyncio.wait_for(gateway.generate_plan(self.context), 0.5)
        self.assertEqual(result.status, 'TIMEOUT')
        gateway = self.gateway(hang)
        pending = asyncio.create_task(gateway.generate_plan(self.context))
        await asyncio.sleep(0.02)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(gateway._busy)
    async def test_global_budget_includes_grounding(self):
        gateway = self.gateway(response({'candidates': []}), self.ok(), config={'max_requests_per_task': 1})
        first = await gateway.ground_target(self.context, 'cabinet_handle_1', [TestCrop()])
        self.assertEqual(first.status, 'NO_DETECTION')
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(len(self.transport.requests), 1)
    async def test_replan_budget_cannot_be_bypassed_by_false_flag(self):
        gateway = self.gateway(self.ok(), self.ok(), config={'max_task_replans': 1})
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'OK')
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'OK')
        result = await gateway.generate_plan(self.context, replan=False)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertEqual(len(self.transport.requests), 2)
    async def test_goal_cannot_change_under_same_task_id(self):
        gateway = self.gateway(self.ok())
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'OK')
        self.context['goal']['args']['target_value'] = 0.5
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'INVALID_OUTPUT')
        self.assertEqual(len(self.transport.requests), 1)
    async def test_grounding_before_the_goal_is_settled_does_not_pin_the_goal(self):
        # Intake observes poses under the draft goal, binds the real one, then plans: only planning pins it.
        gateway = self.gateway(response({'candidates': []}), self.ok())
        draft = copy.deepcopy(self.context)
        draft['goal']['args']['target_value'] = 0.5
        self.assertEqual((await gateway.ground_target(draft, 'cabinet_handle_1', [TestCrop()])).status, 'NO_DETECTION')
        self.assertEqual((await gateway.generate_plan(self.context)).status, 'OK')
    async def test_oversized_input_and_image_policy_fail_before_egress(self):
        gateway = self.gateway()
        result = await gateway.generate_plan(self.context, task_text='x' * 13000)
        self.assertEqual(result.status, 'INVALID_OUTPUT')
        for images in [[TestCrop(face=True)], [TestCrop(size=200001)], [TestCrop(edge=641)], [TestCrop(), TestCrop()], [TestCrop()] * 3]:
            gateway = self.gateway()
            self.assertEqual((await gateway.generate_plan(self.context, images=images)).status, 'INVALID_OUTPUT')
            self.assertEqual(self.transport.requests, [])
    async def test_grounding_identity_box_and_query(self):
        candidate = {'image_id': 'image-1', 'entity_id': 'cabinet_handle_1', 'label': 'handle',
                     'box_xyxy_normalized': [0.1, 0.2, 0.8, 0.9], 'confidence': 0.9}
        gateway = self.gateway(response({'candidates': [candidate]}))
        result = await gateway.ground_target(self.context, 'cabinet_handle_1', [TestCrop()], query='the lower handle', request_id='ground-1')
        self.assertEqual(result.status, 'OK')
        self.assertEqual(result.candidates, (candidate,))
        self.assertEqual(result.request_id, 'ground-1')
        request = self.transport.requests[0]
        self.assertIn('the lower handle', request['input'][0]['content'][0]['text'])
        self.assertEqual(request['input'][0]['content'][1]['type'], 'input_image')
        for field, value in [('entity_id', 'other'), ('image_id', 'missing'), ('box_xyxy_normalized', [0.8, 0.2, 0.1, 0.9])]:
            bad = {**candidate, field: value}
            gateway = self.gateway(response({'candidates': [bad]}), config={'max_plan_regenerations': 0})
            result = await gateway.ground_target(self.context, 'cabinet_handle_1', [TestCrop()])
            self.assertEqual(result.status, 'INVALID_OUTPUT')
            self.assertEqual(result.candidates, ())
    async def test_grounding_requires_known_target_and_image(self):
        gateway = self.gateway()
        self.assertEqual((await gateway.ground_target(self.context, 'missing', [TestCrop()])).status, 'INVALID_OUTPUT')
        self.assertEqual((await gateway.ground_target(self.context, 'cabinet_handle_1', [])).status, 'NEED_OBSERVATION')
        self.assertEqual(self.transport.requests, [])
    async def test_busy_gateway_does_not_queue_another_cloud_wait(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        async def hold():
            entered.set()
            await release.wait()
            return self.ok()
        gateway = self.gateway(hold)
        pending = asyncio.create_task(gateway.generate_plan(self.context))
        await asyncio.wait_for(entered.wait(), 0.5)
        second = await gateway.generate_plan(self.context)
        self.assertEqual(second.status, 'UNAVAILABLE')
        release.set()
        self.assertEqual((await pending).status, 'OK')
    async def test_wrong_provider_model_is_never_used_as_fallback(self):
        wrong = {**self.ok(), 'model': 'different-model'}
        gateway = self.gateway(wrong)
        result = await gateway.generate_plan(self.context)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertIsNone(result.plan)
    async def test_unresponsive_supervisor_callback_is_bounded(self):
        gateway = self.gateway(self.ok())
        async def hang():
            await asyncio.sleep(10)
            return True
        gateway.hold_assertion = hang
        result = await asyncio.wait_for(gateway.generate_plan(self.context), .5)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertEqual(self.transport.requests, [])

    async def test_supervisor_ignoring_timeout_cannot_hold_cloud_gate(self):
        gateway = self.gateway(self.ok())
        release = asyncio.Event()
        cancelled = asyncio.Event()
        finished = asyncio.Event()

        async def stubborn():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            finally:
                finished.set()
            return True

        gateway.hold_assertion = stubborn
        pending = asyncio.create_task(gateway.generate_plan(self.context))
        try:
            done, _ = await asyncio.wait({pending}, timeout=0.4)
            self.assertIn(pending, done, 'A revoked assertion retained the cloud gate')
            self.assertEqual(pending.result().status, 'UNAVAILABLE')
            self.assertFalse(gateway._busy)
            self.assertEqual(self.transport.requests, [])
            await asyncio.wait_for(cancelled.wait(), 0.1)
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 0.5)
            await asyncio.gather(pending, return_exceptions=True)

    async def test_external_cancel_during_stubborn_supervisor_is_prompt(self):
        gateway = self.gateway(self.ok())
        entered = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        callback_errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))

        async def stubborn():
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
                raise RuntimeError('late assertion failure must be consumed')
            finally:
                finished.set()

        gateway.hold_assertion = stubborn
        pending = asyncio.create_task(gateway.generate_plan(self.context))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            pending.cancel()
            done, _ = await asyncio.wait({pending}, timeout=0.2)
            self.assertIn(pending, done, 'Cancellation waited for the revoked assertion')
            self.assertTrue(pending.cancelled())
            self.assertFalse(gateway._busy)
            self.assertEqual(self.transport.requests, [])
            release.set()
            await asyncio.wait_for(finished.wait(), 0.5)
            await asyncio.sleep(0)
            self.assertEqual(callback_errors, [])
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            loop.set_exception_handler(previous_handler)

    async def test_hold_rejection_consumes_racing_provider_failure_privately(self):
        async def provider_failure():
            raise RuntimeError('private provider detail must not be logged')

        gateway = self.gateway(provider_failure)
        checks = 0
        callback_errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))

        async def lose_hold_during_request():
            nonlocal checks
            checks += 1
            if checks >= 3:
                await asyncio.sleep(0.01)
                return False
            return True

        gateway.hold_assertion = lose_hold_during_request
        try:
            result = await gateway.generate_plan(self.context)
            self.assertEqual(result.status, 'UNAVAILABLE')
            self.assertNotIn('private provider', result.detail)
            self.assertEqual(len(self.transport.requests), 1)
            await asyncio.sleep(0)
            self.assertEqual(callback_errors, [])
        finally:
            loop.set_exception_handler(previous_handler)

class SDKConfigurationTests(unittest.TestCase):
    def test_schema_translation_preserves_nonempty_unicode_strings_and_identity(self):
        for original in ({'type': 'string', 'minLength': 1}, {'type': 'integer', 'const': 1}):
            converted = _provider_schema(original)
            self.assertNotIn('const', converted)
            self.assertNotIn('minLength', converted)
            for value in ('', 'a', '\n', '\u00e9', '\U0001f600', 0, 1, True, None):
                accepted = []
                for schema in (original, converted):
                    try:
                        validate_schema(value, schema)
                        accepted.append(True)
                    except ContractError:
                        accepted.append(False)
                self.assertEqual(accepted[0], accepted[1], (original, value))
        with self.assertRaises(ContractError):
            _provider_schema({'type': 'string', 'unrecognizedConstraint': 1})

    def test_sdk_retry_and_model_lock(self):
        with patch('openai.AsyncOpenAI') as sdk:
            OpenAIResponsesTransport(api_key='test-not-a-secret', timeout_s=10)
            sdk.assert_called_once_with(timeout=10, max_retries=0, api_key='test-not-a-secret')
        for config in [{'model': 'anything-else'}, {'store': True}, {'max_requests_per_task': 13}, {'request_timeout_s': float('nan')}, {'max_images': -1}]:
            with self.assertRaises(ValueError):
                AstraReasoner(Catalog(ROOT), config, hold_assertion=lambda: True, epoch_getter=lambda task: 1)
