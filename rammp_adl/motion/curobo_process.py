"""Retained, isolated cuRobo planning process. No robot or ROS transport.

The caller owns skill/planner leasing and independent trajectory admission.
This process serializes requests, pins model assets, and discards responses after
timeouts or protocol faults. It never promotes a candidate to a motion permit.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import select
import subprocess
import threading
import time
import uuid

from ..contracts import Catalog, canonical_json, digest, strict_loads
from .curobo import CUROBO_VERSION, RAMMP_COMMIT
from .curobo_worker import parse_request
from .rolling import MotionError
from .gpu_lease import GpuLeaseJournal


REVIEWED_GPU_IMAGE = "sha256:b34c1bcf9fc094ecb39e7de3591f4e22112b9713e2892783259a2ac0cc7d949e"
MAX_RESPONSE_BYTES = 64*1024*1024
DOCKER_TIMEOUT_S = 15.
CLIENT_STOP_TIMEOUT_S = 5.
IMAGE_ASSET_ROOT = Path('/usr/local/lib/python3.10/dist-packages/curobo/content/assets')
_UNRESOLVED_PROCESSES = []  # Keep a solver lease if Docker cannot prove teardown.


def _file_digest(path):
    return "sha256:"+hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _audit_worker_assets(model_dir, planner_config, *, load_yaml=None):
    """Run in the worker namespace before loading CUDA or model references.

    YAML is an existing dependency of the pinned GPU image, not a new host
    runtime requirement. Planner/robot/world YAML must come from the explicit
    model bind; referenced robot assets may also use the immutable image.
    """
    if load_yaml is None:
        from yaml import safe_load
        load_yaml = lambda path: safe_load(Path(path).read_text())
    root = Path(model_dir).resolve()
    evidence = {}

    def require_path(raw, *, image_allowed=False, directory=False):
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise MotionError('Worker model inputs require explicit absolute asset paths')
        path = Path(raw).resolve()
        if not path.is_relative_to(root) and not (image_allowed and path.is_relative_to(IMAGE_ASSET_ROOT)):
            raise MotionError('Worker model input is outside read-only model/image assets')
        if not (path.is_dir() if directory else path.is_file()):
            raise MotionError('Worker model input is unavailable')
        if not directory:
            evidence[str(path)] = _file_digest(path)
        return path

    config_path = require_path(str(planner_config))
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise MotionError('Worker planner configuration must be a mapping')
    robot_path = require_path(config.get('robot'))
    require_path(config.get('world'))
    robot = load_yaml(robot_path)
    if not isinstance(robot, dict) or not isinstance(robot.get('robot_cfg'), dict):
        raise MotionError('Worker robot configuration is malformed')
    kin = robot['robot_cfg'].get('kinematics')
    if not isinstance(kin, dict):
        raise MotionError('Worker robot kinematics configuration is malformed')
    require_path(kin.get('urdf_path'), image_allowed=True)
    require_path(kin.get('asset_root_path'), image_allowed=True, directory=True)
    # This composition uses the explicit URDF and inlined collision model.
    # Extra config-file resolution must gain its own audited path contract.
    if isinstance(kin.get('collision_spheres'), str) or kin.get('use_usd_kinematics'):
        raise MotionError('Worker requires inlined collision spheres and URDF kinematics')
    return evidence


def _worker_main():
    """Audit read-only input dependencies inside the same isolated container."""
    import sys
    from .curobo_worker import main
    model_dir, config_path = sys.argv[1:3]
    evidence = _audit_worker_assets(model_dir, config_path)
    print(json.dumps({'read_only_worker_inputs':evidence}), file=sys.stderr, flush=True)
    raise SystemExit(main(sys.argv[3:]))


class WarmGpuPlanner:
    """Own one GPU-only container and reuse its initialized planner.

    All calls are synchronous here so async compositions must use to_thread and
    drain cancelled work before releasing the caller's planner lease. A process
    timeout destroys the session; there is no controller/planner fallback.
    """
    def __init__(self, *, planner_config, model_dir, wrapper_checkout, gpu_cache,
                 output, startup_timeout_s=180., request_timeout_s=120.):
        for v in (startup_timeout_s, request_timeout_s):
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 600:
                raise MotionError("Planner deadlines must be finite, positive and at most 600 seconds")
        self.root = Catalog().root
        self.model_dir, self.wrapper = Path(model_dir).resolve(), Path(wrapper_checkout).resolve()
        self.planner_config = Path(planner_config).resolve()
        self.output, self.cache = Path(output).resolve(), Path(gpu_cache).resolve()
        if self.model_dir == Path('/') or not self.planner_config.is_relative_to(self.model_dir):
            raise MotionError("Planner configuration must be inside an explicit model directory")
        if self.output.is_relative_to(self.model_dir) or self.cache.is_relative_to(self.model_dir):
            raise MotionError('Mutable planner output/cache must be outside immutable model assets')
        readonly_sources = (self.model_dir, self.wrapper, *(self.root/n for n in ('rammp_adl','skills','schemas','config','tools')))
        if any(p.is_relative_to(self.cache) or self.cache.is_relative_to(p) for p in readonly_sources):
            raise MotionError('Writable GPU cache must not alias read-only model or source assets')
        for p in (self.model_dir, self.wrapper, self.output, self.cache):
            if ':' in str(p):
                raise MotionError("Docker bind paths may not contain colons")
        self.startup_timeout_s, self.request_timeout_s = startup_timeout_s, request_timeout_s
        self.name = 'rammp-warm-planner-'+uuid.uuid4().hex[:12]
        self._mutex, self._responses = threading.Lock(), queue.Queue(maxsize=1)
        self._closing = threading.Event()
        self._process = self._lease = self._log = self._reader = None
        self._next_id = 1
        self.ready = None
        self._pins = {}
        self._fault = ''
        self._container_may_exist = False
        self._planner_identity = None
        self._teardown_evidence = None
        self._journal = GpuLeaseJournal(self.cache)

    def _read_responses(self):
        pending = bytearray()
        try:
            fd = self._process.stdout.fileno()
            os.set_blocking(fd, False)
            while not self._closing.is_set():
                if not select.select([fd], [], [], .1)[0]:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise MotionError('GPU planner process exited')
                pending.extend(chunk)
                while b'\n' in pending:
                    line, _, pending = pending.partition(b'\n')
                    value = strict_loads(bytes(line), max_bytes=MAX_RESPONSE_BYTES)
                    self._responses.put(value, timeout=.5)
                if len(pending) > MAX_RESPONSE_BYTES:
                    raise MotionError('GPU planner protocol frame exceeds response bound')
        except BaseException as exc:
            try:
                self._responses.put(exc, timeout=.5)
            except queue.Full:
                self._fault = 'Unsolicited GPU protocol frames exceeded the response bound'

    def _receive(self, timeout):
        try:
            if timeout <= 0:
                raise queue.Empty
            response = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise MotionError('GPU planner response deadline exceeded') from exc
        if isinstance(response, BaseException):
            raise MotionError('GPU planner output unavailable: '+str(response)) from response
        if (self._fault or not isinstance(response, dict)
                or type(response.get('protocol')) is not int or response.get('protocol') != 1):
            raise MotionError(self._fault or 'GPU planner protocol mismatch')
        return response

    def _asset_inventory(self):
        if self.model_dir.is_symlink() or not self.model_dir.is_dir():
            raise MotionError('Planner model directory changed or is unavailable')
        current = {}
        for p in sorted(self.model_dir.rglob('*')):
            if p.is_symlink() or not (p.is_dir() or p.is_file()):
                raise MotionError('Planner model assets must be regular files/directories, without symlinks')
            if p.is_file():
                current[str(p)] = _file_digest(p)
        return current

    def _check_assets(self):
        current = self._asset_inventory()
        if current != self._pins:
            raise MotionError('Planner model assets changed during the retained session')

    def _write_request(self, encoded, deadline):
        """Never block on a worker that stops consuming stdin, including close."""
        fd = self._process.stdin.fileno()
        os.set_blocking(fd, False)
        remaining = memoryview(encoded)
        while remaining:
            timeout = deadline-time.monotonic()
            if timeout <= 0 or not select.select([], [fd], [], timeout)[1]:
                raise MotionError('GPU planner request write deadline exceeded')
            try:
                size = os.write(fd, remaining)
            except BlockingIOError:
                continue
            if size <= 0:
                raise MotionError('GPU planner request pipe closed')
            remaining = remaining[size:]

    def _validate_result_provenance(self, result, request):
        if result['status'] != 'planned':
            return
        expected = {'request_digest':digest(request), 'world_identity':request['world_identity'],
                    'planner_source_commit':RAMMP_COMMIT, 'curobo_version':CUROBO_VERSION,
                    'planner_config_digest':self._pins[str(self.planner_config)]}
        if any(result.get(key) != value for key,value in expected.items()):
            raise MotionError('GPU planner result provenance does not match its request or configuration')
        if (result.get('curobo_planned') is not True or result.get('hardware_validated') is not False
                or result.get('independent_validation_required') is not True):
            raise MotionError('GPU planner result must remain an independently unvalidated candidate')
        robot = result.get('planner_robot_config')
        if (not isinstance(robot, dict) or result.get('planner_robot_config_digest') != digest(robot)
                or not isinstance(robot.get('kinematics'), dict)):
            raise MotionError('GPU planner robot configuration provenance is malformed')
        kin = robot['kinematics']
        for field in ('urdf_path', 'asset_root_path'):
            raw = kin.get(field)
            if not isinstance(raw, str) or not Path(raw).is_absolute() or '..' in Path(raw).parts:
                raise MotionError('GPU planner requires explicit read-only robot asset paths')
            path = Path(raw)
            if path.is_relative_to(self.model_dir):
                if field == 'urdf_path' and result.get('planner_urdf_digest') != self._pins.get(str(path)):
                    raise MotionError('GPU planner URDF differs from its pinned model asset')
            elif not path.is_relative_to(IMAGE_ASSET_ROOT):
                raise MotionError('GPU planner robot assets are outside the pinned model/image')
        urdf_digest = result.get('planner_urdf_digest')
        if (not isinstance(urdf_digest, str) or len(urdf_digest) != 71 or not urdf_digest.startswith('sha256:')
                or any(c not in '0123456789abcdef' for c in urdf_digest[7:])):
            raise MotionError('GPU planner URDF digest is malformed')
        identity = (result['planner_robot_config_digest'], urdf_digest, result.get('base_frame'), result.get('ee_link'))
        if (identity[2:] != (kin.get('base_link'), kin.get('ee_link'))
                or any(not isinstance(v, str) or not v for v in identity[2:])
                or (self._planner_identity is not None and identity != self._planner_identity)):
            raise MotionError('GPU retained planner model or frame identity changed')
        self._planner_identity = identity

    def start(self):
        with self._mutex:
            if self.ready is not None or self._process is not None or self._closing.is_set():
                raise MotionError('Planner sessions cannot be restarted or initialized twice')
            self.output.mkdir(parents=True, exist_ok=False)
            self.cache.mkdir(parents=True, exist_ok=True)
            self._pins = self._asset_inventory()
            if str(self.planner_config) not in self._pins or not self.wrapper.is_dir():
                raise MotionError('Planner configuration/model assets or wrapper checkout unavailable')
            self._lease = (self.cache/'rammp-commissioning-planner.lock').open('a')
            try:
                fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lease.close(); self._lease = None
                raise MotionError('This commissioning planner is already leased') from exc
            try:
                self._journal.reconcile()
                info = json.loads(subprocess.run(['docker','image','inspect',REVIEWED_GPU_IMAGE],
                    check=True, capture_output=True, text=True, timeout=15).stdout)[0]
                if (info['Id'] != REVIEWED_GPU_IMAGE or info['Config']['Entrypoint'] != ['python3']
                        or info['Config'].get('Volumes')):
                    raise MotionError('Unexpected GPU image provenance or startup configuration')
                command = ['docker','run','--rm','-i','--name',self.name,'--network','none','--runtime','nvidia',
                    '--read-only','--tmpfs','/tmp:rw,size=2g','--entrypoint','python3',
                    '--user',str(os.getuid())+':'+str(os.getgid()),'-e','PYTHONDONTWRITEBYTECODE=1',
                    '-e','XDG_CACHE_HOME=/gpu-cache','-e','TORCH_EXTENSIONS_DIR=/gpu-cache/torch_extensions',
                    '-e','PYTHONPATH=/workspace:/opt/pinned-wrapper/core',
                    '-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory',
                    '-e','GIT_CONFIG_VALUE_0=/opt/pinned-wrapper']
                bindings = [(self.wrapper,'/opt/pinned-wrapper','ro'),(self.model_dir,str(self.model_dir),'ro'),
                            (self.cache,'/gpu-cache','rw')]
                bindings.extend((self.root/n,'/workspace/'+n,'ro') for n in ('rammp_adl','skills','schemas','config'))
                bindings.append((self.root/'tools/check_design.py','/workspace/tools/check_design.py','ro'))
                for source,target,mode in bindings:
                    if not source.exists(): raise FileNotFoundError(source)
                    command.extend(['-v',str(source)+':'+target+':'+mode])
                command.extend([REVIEWED_GPU_IMAGE,'-c',
                                'from rammp_adl.motion.curobo_process import _worker_main; _worker_main()',
                                str(self.model_dir),str(self.planner_config),'--serve',
                                '--wrapper-source','/opt/pinned-wrapper','--planner-config',str(self.planner_config)])
                (self.output/'invocation.json').write_text(json.dumps({
                    'command':command,'model_assets':self._pins,'robot_commands':False,'network':'none'},indent=2)+'\n')
                self._log = (self.output/'worker.log').open('wb')
                self._journal.reserve(self.name)
                self._container_may_exist = True
                self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 stderr=self._log, bufsize=0,
                                                 pass_fds=(self._lease.fileno(),))
                self._reader = threading.Thread(target=self._read_responses, daemon=True)
                self._reader.start()
                ready = self._receive(self.startup_timeout_s)
                if (ready.get('status') != 'ready' or ready.get('hardware_commands') is not False
                        or ready.get('planner_source_commit') != RAMMP_COMMIT
                        or ready.get('curobo_version') != CUROBO_VERSION
                        or ready.get('planner_config_digest') != self._pins[str(self.planner_config)]):
                    raise MotionError('GPU planner initialization provenance mismatch')
                self._check_assets()
                self.ready = ready
                (self.output/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
                return ready
            except BaseException:
                self._close()
                raise

    def plan(self, request):
        parse_request(request)  # Reject before any expensive process interaction.
        encoded_request = strict_loads(canonical_json(request), max_bytes=1048576)
        with self._mutex:
            if self.ready is None or self._closing.is_set() or self._fault:
                raise MotionError('GPU planner session is unavailable or faulted')
            request_id = self._next_id
            try:
                self._check_assets()
                frame = {'request_id':request_id, 'request':encoded_request}
                encoded = (canonical_json(frame)+'\n').encode()
                if len(encoded) > 1048576: raise MotionError('GPU request envelope exceeds one MiB')
                (self.output/f'request-{request_id}.json').write_bytes(encoded)
                began = time.monotonic()
                deadline = began+self.request_timeout_s
                self._write_request(encoded, deadline)
                response = self._receive(deadline-time.monotonic())
                if (type(response.get('request_id')) is not int or response.get('request_id') != request_id
                        or set(response) != {'protocol','request_id','result'}):
                    raise MotionError('GPU response request identity mismatch')
                result = response['result']
                if (not isinstance(result, dict) or result.get('hardware_commands') is not False
                        or result.get('status') not in ('planned', 'rejected')):
                    raise MotionError('GPU planner response is malformed')
                self._validate_result_provenance(result, encoded_request)
                self._check_assets()
                (self.output/f'response-{request_id}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                with (self.output/'timing.jsonl').open('a') as log:
                    log.write(json.dumps({'request_id':request_id,'round_trip_s':time.monotonic()-began})+'\n')
                self._next_id += 1
                return result
            except BaseException as exc:
                self._fault = str(exc)
                self._close()
                raise

    def _close(self):
        self._closing.set()
        try:
            self._stop_process()
        except BaseException:
            if self._lease is not None and self not in _UNRESOLVED_PROCESSES:
                _UNRESOLVED_PROCESSES.append(self)
            raise
        if self._log is not None: self._log.close()
        if self._lease is not None:
            self._lease.close(); self._lease = None
        if self in _UNRESOLVED_PROCESSES:
            _UNRESOLVED_PROCESSES.remove(self)

    def _docker(self, arguments):
        try:
            return subprocess.run(['docker', *arguments], capture_output=True, text=True, timeout=DOCKER_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None

    def _container_stopped(self):
        """Require a successful daemon response, never infer from CLI failure.

        Filter only this unique owned name. A successful empty list proves
        absence; an inspect error (including 'not found') alone proves nothing.
        Created/restarting/paused containers do not establish quiescence.
        """
        status = self._docker(['ps', '-a', '--filter', 'name=^/'+self.name+'$',
                               '--format', '{{.Names}}\t{{.State}}'])
        if status is not None and status.returncode == 0:
            rows = status.stdout.strip().splitlines()
            if not rows:
                self._teardown_evidence = {'container_name':self.name,'evidence':'successful_exact_name_ps',
                                           'state':'absent','observed_at_monotonic_s':time.monotonic()}
                return True
            if len(rows) == 1:
                fields = rows[0].split('\t')
                if fields == [self.name, 'exited'] or fields == [self.name, 'dead']:
                    self._teardown_evidence = {'container_name':self.name,'evidence':'successful_exact_name_ps',
                                               'state':fields[1],'observed_at_monotonic_s':time.monotonic()}
                    return True
        inspection = self._docker(['container', 'inspect', '--format', '{{json .State}}', self.name])
        if inspection is not None and inspection.returncode == 0:
            try:
                state = strict_loads(inspection.stdout, max_bytes=65536)
                stopped = (isinstance(state, dict) and state.get('Running') is False
                        and state.get('Restarting') is False and type(state.get('Pid')) is int
                        and state['Pid'] == 0 and state.get('Status') in ('exited', 'dead'))
                if stopped:
                    self._teardown_evidence = {'container_name':self.name,'evidence':'successful_named_inspect',
                                               'state':state,'observed_at_monotonic_s':time.monotonic()}
                return stopped
            except (ValueError, RuntimeError):
                pass
        return False

    def _stop_process(self):
        if self._process is not None:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=CLIENT_STOP_TIMEOUT_S)
            except (OSError, subprocess.TimeoutExpired):
                self._docker(['rm', '-f', self.name])
                # The Docker client can hang even when the GPU container has
                # stopped. Reap only our client, then independently ask Docker.
                if self._process.poll() is None:
                    self._process.kill()
                self._process.wait(timeout=CLIENT_STOP_TIMEOUT_S)
            if self._reader is not None: self._reader.join(timeout=1.)
            self._process.stdout.close()
        if self._container_may_exist:
            if not self._container_stopped():
                self._docker(['rm', '-f', self.name])
                if not self._container_stopped():
                    raise MotionError('GPU container teardown is unproven; planner lease retained')
            # The durable lease protects later processes as well as this flock.
            # Remove only our current session's container, never an old journal's.
            if self._teardown_evidence.get('state') != 'absent':
                self._docker(['rm', '-f', self.name])
            self._journal.release(self.name)
            self._container_may_exist = False
            (self.output/'teardown.json').write_text(json.dumps(self._teardown_evidence,indent=2)+'\n')

    def close(self):
        with self._mutex:
            # Retry a previous unresolved teardown when daemon access returns.
            if not self._closing.is_set() or self._lease is not None:
                self._close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
