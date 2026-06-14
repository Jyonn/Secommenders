import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARTIFACT_ROOT = ROOT / 'artifacts'
SCHEDULER_ROOT = ARTIFACT_ROOT / 'scheduler'
TRAINED_ROOT = ARTIFACT_ROOT / 'trained'


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(text: str):
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', text).strip('._-') or 'exp'


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def run_dir_completed(run_dir: Path):
    meta = read_json_if_exists(run_dir / 'meta.json')
    return isinstance(meta, dict) and 'test_metrics' in meta


def looks_like_tqdm_progress(line: str):
    text = str(line).strip()
    if not text:
        return False
    if text.count('|') < 2:
        return False
    return any(token in text for token in ('%|', 'it/s', 's/it'))


def collapse_tqdm_progress(text: str):
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    collapsed = []
    pending_progress = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if looks_like_tqdm_progress(line):
            pending_progress = line
            continue
        if pending_progress is not None:
            collapsed.append(pending_progress)
            pending_progress = None
        collapsed.append(line)
    if pending_progress is not None:
        collapsed.append(pending_progress)
    return '\n'.join(collapsed)


def read_log_for_report(path: Path, max_bytes: int = 2_000_000):
    if not path.exists():
        return ''
    data = path.read_bytes()
    text = collapse_tqdm_progress(data.decode('utf-8', errors='ignore'))
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode('utf-8', errors='ignore')


def performance_from_meta(meta: dict | None):
    if not isinstance(meta, dict):
        return None
    for key in ('test_metrics', 'valid_metrics', 'checkpoint_valid_metrics'):
        metrics = meta.get(key)
        if isinstance(metrics, dict):
            return metrics
    return None


def infer_status_from_run_dir(run_dir: Path):
    meta = read_json_if_exists(run_dir / 'meta.json') or {}
    if run_dir_completed(run_dir):
        return 'done'
    if meta.get('status') == 'failed' or meta.get('error'):
        return 'failed'
    return None


def infer_seed(configuration: dict | None):
    if not isinstance(configuration, dict):
        return 42
    logical = configuration.get('logical_train_args')
    if isinstance(logical, dict) and logical.get('seed') is not None:
        return int(logical['seed'])
    base = configuration.get('base_args')
    if isinstance(base, dict) and base.get('seed') is not None:
        return int(base['seed'])
    return 42


class HttpClient:
    def __init__(self, uri: str, auth: str = ''):
        self.uri = uri.rstrip('/')
        self.auth = auth

    def get_json(self, path: str, query: dict | None = None):
        url = f'{self.uri}{path}'
        if query:
            url = f'{url}?{urlencode(query)}'
        headers = {'Authentication': self.auth} if self.auth else {}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()


@dataclass
class ApiReply:
    identifier: str
    body: object = None
    msg: str = ''
    code: int = 0
    http_code: int = 0

    @property
    def ok(self):
        return self.identifier == 'OK'


class Server:
    ENV_URI = 'SECOMMENDER_BACKEND_URI'
    ENV_AUTH = 'SECOMMENDER_BACKEND_AUTH_TOKEN'

    def __init__(self, uri: str, auth: str):
        self.uri = uri.rstrip('/')
        self.auth = auth

    def _headers(self):
        return {'Authentication': self.auth}

    def _request(self, method: str, uri: str, **kwargs):
        response = requests.request(method, uri, headers=self._headers(), timeout=30, **kwargs)
        payload = response.json()
        return ApiReply(
            identifier=str(payload.get('identifier') or ''),
            body=payload.get('body'),
            msg=str(payload.get('msg') or ''),
            code=int(payload.get('code') or response.status_code),
            http_code=response.status_code,
        )

    def post(self, uri: str, data: dict):
        return self._request('post', uri, json=data)

    def put(self, uri: str, data: dict):
        return self._request('put', uri, json=data)

    def create_or_get_evaluation(self, signature: str, command: str, configuration: str, name: str = ''):
        return self.post(
            f'{self.uri}/evaluations/',
            {
                'signature': signature,
                'command': command,
                'configuration': configuration,
                'name': name,
            },
        )

    def create_or_get_experiment(self, signature: str, seed: int):
        return self.post(
            f'{self.uri}/experiments/',
            {
                'signature': signature,
                'seed': seed,
            },
        )

    def register_experiment(
        self,
        session: str,
        *,
        pid=None,
        hostname: str = '',
        run_dir: str = '',
        log_path: str = '',
        command: str = '',
        phase: str = '',
    ):
        return self.post(
            f'{self.uri}/experiments/{session}/register',
            {
                'pid': pid,
                'hostname': hostname,
                'run_dir': run_dir,
                'log_path': log_path,
                'command': command,
                'phase': phase,
            },
        )

    def update_experiment(
        self,
        session: str,
        *,
        status: str,
        phase: str = '',
        meta=None,
        performance=None,
        log: str = '',
        error: str = '',
    ):
        return self.put(
            f'{self.uri}/experiments/',
            {
                'session': session,
                'status': status,
                'phase': phase,
                'meta': meta,
                'performance': performance,
                'log': log,
                'error': error,
            },
        )


class StateFile:
    def __init__(self, path: Path):
        self.path = path
        self.payload = json.loads(path.read_text())
        self.modified = False

    def save(self):
        if not self.modified:
            return
        self.path.write_text(json.dumps(self.payload, indent=2) + '\n')
        self.modified = False


@dataclass
class LocalExperiment:
    state: StateFile
    index: int
    exp: dict

    @property
    def name(self):
        return str(self.exp.get('name') or f'exp{self.index:03d}')

    @property
    def plan_name(self):
        return str(self.state.payload.get('name') or self.state.path.parent.name)

    @property
    def status(self):
        return str(self.exp.get('status') or '')

    @property
    def phase(self):
        return str(self.exp.get('phase') or '')

    @property
    def signature(self):
        return str(self.exp.get('report_signature') or '')

    @property
    def seed(self):
        return int(self.exp.get('report_seed') or 42)

    @property
    def session(self):
        value = self.exp.get('report_session')
        return str(value) if value else None

    @property
    def run_dir(self):
        value = self.exp.get('run_dir')
        return Path(value) if value else None

    @property
    def log_path(self):
        value = self.exp.get('log_path')
        return Path(value) if value else None

    @property
    def command(self):
        return str(self.exp.get('report_command') or '')

    @property
    def configuration(self):
        return str(self.exp.get('report_configuration') or '')

    @property
    def uploaded_at(self):
        value = self.exp.get('report_uploaded_at')
        return str(value) if value else None

    @property
    def last_error(self):
        return str(self.exp.get('last_error') or '')

    @property
    def terminal(self):
        return self.status in {'done', 'failed'}

    @property
    def run_id(self):
        return self.run_dir.name if self.run_dir else ''

    def set_session(self, session: str):
        if self.exp.get('report_session') == session:
            return
        self.exp['report_session'] = session
        self.state.modified = True

    def mark_uploaded(self):
        self.exp['report_uploaded_at'] = utc_now_iso()
        self.exp['report_upload_error'] = None
        self.state.modified = True

    def mark_upload_error(self, error: str):
        self.exp['report_upload_error'] = error
        self.state.modified = True


@dataclass
class ArtifactCandidate:
    signature: str
    seed: int
    name: str
    command: str
    configuration: str
    status: str
    phase: str
    run_dir: Path
    log_path: Path
    last_error: str
    session: Optional[str] = None


@dataclass
class RepairAction:
    key: tuple[str, int]
    source: str
    reason: str
    session: Optional[str]
    local: Optional[LocalExperiment] = None
    artifact: Optional[ArtifactCandidate] = None


def state_paths_from_args(args):
    if args.state:
        return [Path(path).resolve() for path in args.state]

    if args.plan:
        paths = []
        for raw_path in args.plan:
            plan_path = Path(raw_path).resolve()
            plan = yaml.safe_load(plan_path.read_text()) or {}
            plan_name = sanitize_name(plan.get('name') or plan_path.stem)
            state_path = ROOT / 'artifacts' / 'scheduler' / plan_name / 'state.json'
            paths.append(state_path)
        return paths

    return sorted(SCHEDULER_ROOT.glob('*/state.json'))


def load_local_experiments(state_paths: list[Path]):
    states = []
    experiments = []
    for path in state_paths:
        if not path.exists():
            print(f'warning: state file not found: {path}')
            continue
        state = StateFile(path)
        states.append(state)
        for index, exp in enumerate(state.payload.get('experiments', []), start=1):
            local = LocalExperiment(state=state, index=index, exp=exp)
            if not local.signature:
                continue
            experiments.append(local)
    return states, experiments


def parse_backend_from_plan(plan_path: Path):
    plan = yaml.safe_load(plan_path.read_text()) or {}
    backend = plan.get('backend') or {}
    uri = backend.get('uri')
    auth = backend.get('auth_token')
    uri_env = backend.get('uri_env')
    auth_env = backend.get('auth_env')
    if uri_env:
        uri = os.environ.get(uri_env, uri)
    if auth_env:
        auth = os.environ.get(auth_env, auth)
    return uri, auth


def build_clients(args):
    uri = args.uri
    auth = args.auth
    if args.auth_env:
        auth = os.environ.get(args.auth_env, auth)

    if args.plan:
        for raw_path in args.plan:
            plan_uri, plan_auth = parse_backend_from_plan(Path(raw_path).resolve())
            uri = uri or plan_uri
            auth = auth or plan_auth
            if uri and auth:
                break

    uri = uri or os.environ.get(Server.ENV_URI)
    auth = auth or os.environ.get(Server.ENV_AUTH, '')
    if not uri:
        raise ValueError('backend uri is required; use --uri, --plan, or SECOMMENDER_BACKEND_URI')

    http = HttpClient(uri=uri, auth=auth)
    server = Server(uri=uri, auth=auth) if auth else None
    return http, server


def fetch_remote_evaluations(http: HttpClient):
    evaluations = []
    current_page = 1
    total_page = None
    while total_page is None or current_page <= total_page:
        payload = http.get_json('/evaluations/', {'page': current_page})
        if payload.get('identifier') != 'OK':
            raise ValueError(f'failed to fetch evaluations page={current_page}: {payload}')
        body = payload['body']
        evaluations.extend(body['evaluations'])
        total_page = int(body['total_page'])
        current_page += 1
    return evaluations


def fetch_remote_evaluation_detail(http: HttpClient, signature: str):
    payload = http.get_json(f'/evaluations/{signature}')
    if payload.get('identifier') != 'OK':
        raise ValueError(f'failed to fetch evaluation detail for {signature}: {payload}')
    return payload['body']


def build_artifact_candidate(detail: dict, session: str | None):
    data_name = str(detail.get('data_name') or '').strip().lower()
    run_id = str(detail.get('run_id') or '').strip()
    if not data_name or not run_id:
        return None
    run_dir = TRAINED_ROOT / data_name / run_id
    if not run_dir.exists():
        return None
    status = infer_status_from_run_dir(run_dir)
    if status is None:
        return None
    meta = read_json_if_exists(run_dir / 'meta.json') or {}
    configuration = detail.get('configuration') or {}
    return ArtifactCandidate(
        signature=str(detail['signature']),
        seed=infer_seed(configuration),
        name=str(detail.get('name') or detail['signature']),
        command=str(detail.get('command') or ''),
        configuration=json.dumps(configuration, indent=2, sort_keys=True),
        status=status,
        phase=str(meta.get('phase') or 'train'),
        run_dir=run_dir,
        log_path=Path(meta.get('log_path')) if meta.get('log_path') else run_dir / 'train.log',
        last_error=str(meta.get('error') or ''),
        session=session,
    )


def pick_best_local(candidates: list[LocalExperiment], session: str | None):
    if not candidates:
        return None

    def score(item: LocalExperiment):
        return (
            1 if item.terminal else 0,
            1 if session and item.session == session else 0,
            1 if item.run_dir and item.run_dir.exists() else 0,
            1 if item.status == 'done' else 0,
            1 if item.uploaded_at else 0,
        )

    return sorted(candidates, key=score, reverse=True)[0]


def ensure_remote_session(server: Server, exp_like, default_name: str):
    if exp_like.session:
        return exp_like.session
    evaluation_reply = server.create_or_get_evaluation(
        signature=exp_like.signature,
        command=exp_like.command,
        configuration=exp_like.configuration,
        name=default_name,
    )
    if not evaluation_reply.ok:
        raise ValueError(f'failed to create evaluation: {evaluation_reply.msg or evaluation_reply.identifier}')
    experiment_reply = server.create_or_get_experiment(
        signature=exp_like.signature,
        seed=int(exp_like.seed),
    )
    if not experiment_reply.ok:
        raise ValueError(f'failed to create experiment: {experiment_reply.msg or experiment_reply.identifier}')
    session = str(experiment_reply.body)
    if isinstance(exp_like, LocalExperiment):
        exp_like.set_session(session)
    return session


def upload_terminal_experiment(server: Server, exp_like, forced_session: str | None = None):
    session = forced_session or ensure_remote_session(server, exp_like, exp_like.name)
    if isinstance(exp_like, LocalExperiment) and forced_session:
        exp_like.set_session(forced_session)

    run_dir = exp_like.run_dir
    meta = read_json_if_exists(run_dir / 'meta.json') if run_dir else None
    meta = meta or {}
    log_path = exp_like.log_path if exp_like.log_path else (run_dir / 'train.log' if run_dir else None)
    log_text = read_log_for_report(log_path) if log_path and log_path.exists() else ''
    performance = performance_from_meta(meta)
    if len(log_text.encode('utf-8')) >= 2_000_000:
        meta = dict(meta)
        meta['report_log_truncated'] = True

    register_reply = server.register_experiment(
        session,
        pid=meta.get('pid'),
        hostname=str(meta.get('hostname') or ''),
        run_dir=str(run_dir) if run_dir else '',
        log_path=str(meta.get('log_path') or log_path or ''),
        command=str(meta.get('command') or exp_like.command or ''),
        phase=str(exp_like.phase or ''),
    )
    if not register_reply.ok:
        raise ValueError(f'failed to register remote experiment: {register_reply.msg or register_reply.identifier}')

    remote_status = 'completed' if exp_like.status == 'done' else 'failed'
    error_text = ''
    if remote_status == 'failed':
        error_text = str(meta.get('error') or exp_like.last_error or '')
    reply = server.update_experiment(
        session,
        status=remote_status,
        phase=str(exp_like.phase or ''),
        meta=meta,
        performance=performance,
        log=log_text,
        error=error_text,
    )
    if not reply.ok:
        raise ValueError(f'failed to update remote experiment: {reply.msg or reply.identifier}')

    if isinstance(exp_like, LocalExperiment):
        exp_like.mark_uploaded()


def build_actions(http: HttpClient, local_experiments: list[LocalExperiment], remote_evaluations: list[dict]):
    by_signature_seed = {}
    for exp in local_experiments:
        by_signature_seed.setdefault((exp.signature, exp.seed), []).append(exp)

    actions = {}
    unresolved = []
    detail_cache = {}

    def get_detail(signature: str):
        if signature not in detail_cache:
            detail_cache[signature] = fetch_remote_evaluation_detail(http, signature)
        return detail_cache[signature]

    for evaluation in remote_evaluations:
        signature = str(evaluation['signature'])
        experiments = list(evaluation.get('experiments') or [])
        if not experiments:
            detail = get_detail(signature)
            seed = infer_seed(detail.get('configuration') or {})
            locals_for_signature = by_signature_seed.get((signature, seed), [])
            local = pick_best_local(locals_for_signature, session=None)
            key = (signature, seed)
            if local and local.terminal:
                actions[key] = RepairAction(
                    key=key,
                    source='state',
                    reason='remote evaluation has no experiment session',
                    session=None,
                    local=local,
                )
                continue
            artifact = build_artifact_candidate(detail, session=None)
            if artifact:
                actions[key] = RepairAction(
                    key=key,
                    source='artifact',
                    reason='remote evaluation has no experiment session',
                    session=None,
                    artifact=artifact,
                )
            else:
                unresolved.append(f'orphan evaluation signature={signature} name={evaluation.get("name")}')
            continue

        for remote_exp in experiments:
            if str(remote_exp.get('status') or '') != 'created':
                continue
            seed = int(remote_exp.get('seed') or 42)
            session = str(remote_exp.get('session') or '')
            key = (signature, seed)
            local = pick_best_local(by_signature_seed.get(key, []), session=session)
            if local and local.terminal:
                actions[key] = RepairAction(
                    key=key,
                    source='state',
                    reason='remote experiment stuck at created',
                    session=session,
                    local=local,
                )
                continue

            detail = get_detail(signature)
            artifact = build_artifact_candidate(detail, session=session)
            if artifact and artifact.seed == seed:
                actions[key] = RepairAction(
                    key=key,
                    source='artifact',
                    reason='remote experiment stuck at created',
                    session=session,
                    artifact=artifact,
                )
            else:
                unresolved.append(
                    f'created experiment signature={signature} seed={seed} session={session} name={evaluation.get("name")}'
                )

    for exp in local_experiments:
        if not exp.terminal:
            continue
        if exp.uploaded_at:
            continue
        key = (exp.signature, exp.seed)
        if key in actions:
            continue
        actions[key] = RepairAction(
            key=key,
            source='state',
            reason='local terminal experiment not marked uploaded',
            session=exp.session,
            local=exp,
        )

    return list(actions.values()), unresolved


def print_summary(remote_evaluations: list[dict], local_experiments: list[LocalExperiment], actions: list[RepairAction], unresolved: list[str]):
    orphan_count = sum(1 for evaluation in remote_evaluations if not (evaluation.get('experiments') or []))
    created_count = sum(
        1
        for evaluation in remote_evaluations
        for exp in (evaluation.get('experiments') or [])
        if str(exp.get('status') or '') == 'created'
    )
    terminal_local = sum(1 for exp in local_experiments if exp.terminal)
    uploaded_local = sum(1 for exp in local_experiments if exp.uploaded_at)

    print('Remote Report Audit')
    print(f'  remote_evaluations={len(remote_evaluations)}')
    print(f'  remote_orphans={orphan_count}')
    print(f'  remote_created_experiments={created_count}')
    print(f'  local_state_experiments={len(local_experiments)}')
    print(f'  local_terminal_experiments={terminal_local}')
    print(f'  local_uploaded_experiments={uploaded_local}')
    print(f'  repair_candidates={len(actions)}')
    print(f'  unresolved={len(unresolved)}')

    if actions:
        print('\nRepair Candidates')
        for action in actions:
            exp_like = action.local or action.artifact
            print(
                '  '
                f'signature={action.key[0]} seed={action.key[1]} source={action.source} '
                f'status={exp_like.status} reason="{action.reason}" '
                f'session={action.session or exp_like.session or "-"} '
                f'run_dir={exp_like.run_dir}'
            )

    if unresolved:
        print('\nUnresolved')
        for line in unresolved:
            print(f'  {line}')


def apply_actions(server: Server | None, actions: list[RepairAction], limit: int | None):
    if server is None:
        raise ValueError('apply requires backend auth; set --auth/--auth-env/SECOMMENDER_BACKEND_AUTH_TOKEN')

    completed = 0
    failed = 0
    for action in actions[:limit] if limit else actions:
        exp_like = action.local or action.artifact
        label = f'signature={action.key[0]} seed={action.key[1]} source={action.source}'
        print(f'APPLY {label} reason="{action.reason}"')
        try:
            upload_terminal_experiment(server, exp_like, forced_session=action.session)
        except Exception as exc:
            failed += 1
            if action.local:
                action.local.mark_upload_error(repr(exc))
            print(f'  FAILED {label} error={repr(exc)}')
        else:
            completed += 1
            print(f'  OK {label}')
    print(f'\nApply Summary repaired={completed} failed={failed}')


def save_modified_states(states: list[StateFile]):
    for state in states:
        state.save()


def parse_args():
    parser = argparse.ArgumentParser(description='Audit and reconcile Secommenders remote experiment reports.')
    parser.add_argument('--plan', action='append', help='Scheduler plan yaml. Can be passed multiple times.')
    parser.add_argument('--state', action='append', help='Scheduler state.json path. Can be passed multiple times.')
    parser.add_argument('--uri', help='Backend base uri. Defaults to plan backend or SECOMMENDER_BACKEND_URI.')
    parser.add_argument('--auth', help='Backend auth token. Defaults to plan backend or SECOMMENDER_BACKEND_AUTH_TOKEN.')
    parser.add_argument('--auth-env', help='Environment variable name containing backend auth token.')
    parser.add_argument('--apply', action='store_true', help='Actually upload repairs instead of dry-run only.')
    parser.add_argument('--limit', type=int, default=0, help='Only apply the first N repair actions. 0 means no limit.')
    return parser.parse_args()


def main():
    args = parse_args()
    state_paths = state_paths_from_args(args)
    states, local_experiments = load_local_experiments(state_paths)
    http, server = build_clients(args)
    remote_evaluations = fetch_remote_evaluations(http)
    actions, unresolved = build_actions(http, local_experiments, remote_evaluations)
    print_summary(remote_evaluations, local_experiments, actions, unresolved)

    if args.apply:
        apply_actions(server, actions, limit=args.limit or None)
        save_modified_states(states)


if __name__ == '__main__':
    main()
