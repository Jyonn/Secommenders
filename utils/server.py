import os
from typing import Any, Dict, Iterator, Optional

import requests
from pigmento import pnt


class BaseResp:
    def __init__(self, resp: Dict[str, Any], http_code: Optional[int] = None):
        self.msg = resp.get('msg')
        self.identifier = resp.get('identifier')
        self.append_msg = resp.get('append_msg')
        self.debug_msg = resp.get('debug_msg')
        self.code = resp.get('code')
        self.body = resp.get('body')
        self.http_code = resp.get('http_code', http_code)

    @property
    def ok(self) -> bool:
        return self.identifier == 'OK'


class ExperimentBody:
    def __init__(self, body: Dict[str, Any]):
        self.signature = body.get('signature')
        self.session = body.get('session')
        self.seed = body.get('seed')
        self.status = body.get('status')
        self.phase = body.get('phase')
        self.performance = body.get('performance')
        self.meta = body.get('meta')
        self.is_completed = body.get('is_completed')
        self.created_at = body.get('created_at')
        self.started_at = body.get('started_at')
        self.completed_at = body.get('completed_at')
        self.pid = body.get('pid')
        self.error = body.get('error')


class EvaluationBody:
    def __init__(self, body: Dict[str, Any]):
        self.signature = body.get('signature')
        self.name = body.get('name')
        self.command = body.get('command')
        self.configuration = body.get('configuration')
        self.created_at = body.get('created_at')
        self.modified_at = body.get('modified_at')
        self.comment = body.get('comment')
        self.experiments = [ExperimentBody(exp) for exp in body.get('experiments', [])]


class Server:
    ENV_URI = 'SECOMMENDER_BACKEND_URI'
    ENV_AUTH = 'SECOMMENDER_BACKEND_AUTH_TOKEN'
    DEFAULT_TIMEOUT = 30

    def __init__(self, uri: str, auth: str, timeout: int = DEFAULT_TIMEOUT):
        self.uri = uri.rstrip('/')
        self.auth = auth
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        uri = os.environ.get(cls.ENV_URI)
        auth = os.environ.get(cls.ENV_AUTH)
        if not uri or not auth:
            return None
        return cls(uri=uri, auth=auth)

    @staticmethod
    def calculate_bytes(data: Dict[str, Any]) -> int:
        return sum(len(str(key)) + len(str(value)) for key, value in data.items())

    def _headers(self):
        return {'Authentication': self.auth}

    def post(self, uri: str, data: Dict[str, Any]) -> BaseResp:
        pnt(f'uploading {self.calculate_bytes(data)} bytes to {uri}')
        response = requests.post(uri, headers=self._headers(), json=data, timeout=self.timeout)
        return BaseResp(response.json(), http_code=response.status_code)

    def put(self, uri: str, data: Dict[str, Any]) -> BaseResp:
        pnt(f'uploading {self.calculate_bytes(data)} bytes to {uri}')
        response = requests.put(uri, headers=self._headers(), json=data, timeout=self.timeout)
        return BaseResp(response.json(), http_code=response.status_code)

    def delete(self, uri: str) -> BaseResp:
        pnt(f'sending delete request to {uri}')
        response = requests.delete(uri, headers=self._headers(), timeout=self.timeout)
        return BaseResp(response.json(), http_code=response.status_code)

    def get(self, uri: str, query: Dict[str, Any]) -> BaseResp:
        pnt(f'sending query request to {uri} with {query}')
        response = requests.get(uri, headers=self._headers(), params=query, timeout=self.timeout)
        return BaseResp(response.json(), http_code=response.status_code)

    def get_all_evaluations(self) -> Iterator[EvaluationBody]:
        total_page = None
        current_page = 1
        while total_page is None or current_page <= total_page:
            response = self.get(f'{self.uri}/evaluations/', {'page': current_page})
            if not response.ok:
                raise ValueError(f'unable to fetch evaluations: {response.msg or response.identifier}')
            total_page = response.body['total_page']
            for evaluation in response.body['evaluations']:
                yield EvaluationBody(evaluation)
            current_page += 1

    def get_experiment_info(self, session: str) -> BaseResp:
        return self.get(f'{self.uri}/experiments/', {'session': session})

    def create_or_get_evaluation(self, signature: str, command: str, configuration: str, name: str = '') -> BaseResp:
        return self.post(
            f'{self.uri}/evaluations/',
            {
                'signature': signature,
                'command': command,
                'configuration': configuration,
                'name': name,
            },
        )

    def delete_evaluation(self, signature: str) -> BaseResp:
        return self.delete(f'{self.uri}/evaluations/{signature}')

    def create_or_get_experiment(self, signature: str, seed: int) -> BaseResp:
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
        pid: Optional[int] = None,
        hostname: str = '',
        run_dir: str = '',
        log_path: str = '',
        command: str = '',
        phase: str = '',
    ) -> BaseResp:
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
    ) -> BaseResp:
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
