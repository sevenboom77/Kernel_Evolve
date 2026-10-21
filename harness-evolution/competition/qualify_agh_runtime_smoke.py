"""Start an independently built AGH daemon and prompt it against local synthetic HTTP.

This is a delivery startup check, not a replacement for the full SDK tests or
an actual model/GPU search. It reads only the new isolated synthetic credential.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def put(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return ref(path)


def qualify(runtime, node, out):
    runtime, node, out = Path(runtime).resolve(), Path(node).resolve(), Path(out).resolve()
    if out.exists() or not (runtime / 'agnes.mjs').is_file() or not node.is_file():
        raise ValueError('independent runtime, node and fresh output required')
    inventory = [ref(path) for path in sorted(runtime.rglob('*')) if path.is_file()]
    if len(inventory) != 175:
        raise ValueError('expected complete175-file runtime')
    out.mkdir(parents=True, mode=0o700)
    home, work = out / 'isolated-home', out / 'workspace'
    work.mkdir()
    captures, server_errors, checks = [], [], []

    class LocalSynthetic(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path not in ('/v1/messages', '/v1/messages?beta=true'):
                    raise ValueError('unexpected synthetic endpoint')
                body = self.rfile.read(int(self.headers['content-length']))
                document = json.loads(body)
                if document['model'] != 'mimo-v2.6-pro':
                    raise ValueError('unexpected synthetic model identity')
                path = out / f'actual-HTTP-body-{len(captures)+1:02d}.json'
                with path.open('xb') as stream:
                    stream.write(body)
                captures.append(ref(path))
                events = [
                    {'type': 'message_start', 'message': {'id': 'local-smoke', 'type': 'message', 'role': 'assistant',
                        'content': [], 'model': 'mimo-v2.6-pro', 'stop_reason': None, 'stop_sequence': None,
                        'usage': {'input_tokens': 100, 'output_tokens': 0}}},
                    {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                    {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'portable-smoke-ok'}},
                    {'type': 'content_block_stop', 'index': 0},
                    {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 10}},
                    {'type': 'message_stop'},
                ]
                payload = ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:
                server_errors.append(type(error).__name__ + ': ' + str(error))
                self.send_error(500, 'local synthetic fixture refused')

    server = ThreadingHTTPServer(('127.0.0.1', 0), LocalSynthetic)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    route, model = 'local-synthetic', 'mimo-v2.6-pro'
    base_url = f'http://127.0.0.1:{server.server_port}'
    secret = home / 'secrets/synthetic/cpu-only'
    secret.parent.mkdir(parents=True, mode=0o700)
    secret.write_text('CPU_SYNTHETIC_NOT_A_REAL_CREDENTIAL\n')
    secret.chmod(0o600)
    profile = {'name': 'local-dev', 'schemaVersion': 1,
        'provider': {'routes': [{'route': route, 'api': 'anthropic-messages', 'baseUrl': base_url,
            'credentialRef': 'secret://synthetic/cpu-only', 'models': [{'id': model,
                'name': 'Local synthetic startup check', 'api': 'anthropic-messages', 'route': route, 'baseUrl': base_url,
                'reasoning': True, 'thinkingLevelMap': {'off': 'off'}, 'input': ['text', 'image'],
                'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
                'contextWindow': 32768, 'maxTokens': 4096, 'toolCallFormats': ['native'],
                'thinkingReplay': 'native', 'contract_id': None}]}]},
        'limits': {'daemon.max_workers': 10, 'session.max_provider_requests': 24},
        'computerUse': {'enabled': False}, 'presets': {'default': 'standard', 'allowed': ['standard']},
        'packages': [], 'adapters': {'secrets': {'kind': 'file', 'path': str(home / 'secrets')}},
        'policy': {'workspacePackages': 'require-project-trust'}}
    profile_path = home / 'profiles/local-dev/profile.yaml'
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(profile, indent=2))
    env = {key: value for key, value in os.environ.items()
        if not any(part in key.upper() for part in ('TOKEN', 'SECRET', 'CREDENTIAL', 'API_KEY'))
        and not key.startswith(('AGH_', 'AGNES_', 'OPENAI_', 'ANTHROPIC_'))
        and key.lower() not in {'http_proxy', 'https_proxy', 'all_proxy'}}
    env.update(AGH_HOME=str(home), AGNES_PROFILE='local-dev', CUDA_VISIBLE_DEVICES='',
        PATH=str(node.parent) + ':' + env.get('PATH', ''), PYTHONDONTWRITEBYTECODE='1')
    put(out / 'inputs.json', {'actor': ref(__file__), 'node': ref(node), 'runtime_inventory': inventory,
        'isolated_profile': ref(profile_path), 'real_credentials_read': False,
        'external_model_GPU_calls': 0, 'scope': 'daemon and CLI print with loopback-only synthetic model'})

    def run(label, argv):
        start = time.monotonic()
        log = out / (label + '.log')
        with log.open('x') as stream:
            result = subprocess.run(argv, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
        check = {'stage': label, 'command': argv, 'exit_code': result.returncode,
            'elapsed_seconds': time.monotonic() - start, 'log': ref(log)}
        checks.append(check)
        put(out / (label + '-receipt.json'), check)
        print(json.dumps({'stage': label, 'exit_code': result.returncode}), flush=True)
        if result.returncode:
            raise RuntimeError(label + ' failed; inspect retained log')
        return log

    daemon = lambda verb: [str(node), str(runtime / 'agnes.mjs'), 'daemon', verb,
        '--home', str(home), '--profile', 'local-dev', '--data-dir', str(home / 'data'), '--workspace', str(work)]
    code, error = 0, None
    try:
        run('daemon-start', daemon('start'))
        run('daemon-status', daemon('status'))
        log = run('actual-CLI-HTTP-prompt', [str(node), str(runtime / 'agnes.mjs'), '-p',
            'Local synthetic startup check. Answer portable-smoke-ok.', '--mode', 'json',
            '--profile', 'local-dev', '--cwd', str(work), '--model', 'primary='+route+'/'+model])
        if not captures or 'portable-smoke-ok' not in log.read_text() or server_errors:
            raise ValueError('actual local HTTP/CLI output not verified')
    except Exception as exc:
        code, error = 1, type(exc).__name__ + ': ' + str(exc)
    finally:
        if (home / 'data/daemon/owner.json').exists():
            try:
                run('owned-daemon-stop', daemon('stop'))
            except Exception as exc:
                code, error = 1, type(exc).__name__ + ': ' + str(exc)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    remaining = (home / 'data/daemon/owner.json').exists()
    if remaining:
        code, error = 1, 'own daemon owner record remains'
    for row in inventory:
        if ref(row['path']) != row:
            code, error = 1, 'runtime bytes changed during startup check'
    proof = {'status': 'ACTUAL_INDEPENDENT_RUNTIME_DAEMON_CLI_LOCAL_HTTP_SMOKE_PASS' if not code else 'ACTUAL_STARTUP_CHECK_FAILED',
        'inputs': ref(out / 'inputs.json'), 'checks': checks, 'actual_HTTP_bodies': captures,
        'server_errors': server_errors, 'error': error, 'exit_code': code,
        'owned_daemon_owner_absent': not remaining, 'owned_HTTP_server_closed': True,
        'runtime_byte_inventory_unchanged': not code, 'real_credentials_read': False,
        'external_model_GPU_calls': 0, 'full_SDK_requalification_claimed': False,
        'final_ZIP_extract_tested_by_this_check': False, 'RSI_proven': False}
    put(out / 'verification.json', proof)
    print(json.dumps({'proof': ref(out / 'verification.json'), 'actual_HTTP_bodies': len(captures), 'exit_code': code}), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--node', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(qualify(args.runtime, args.node, args.out))
