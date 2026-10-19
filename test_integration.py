"""Tests sans micro : configuration et diffusion vers un vrai Live Server."""

import json
import contextlib
import io
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from configuration import load_settings, publish_url
from main import RemotePublisher, Tee, parse_args


class ConfigurationTests(unittest.TestCase):
    def test_file_environment_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            config = Path(directory) / '.env'
            config.write_text("LIVE_SERVER=http://example.test:8000\nLIVE_API_KEY='secret&+$value'\n", encoding='utf-8')
            self.assertEqual(load_settings(config)['LIVE_API_KEY'], 'secret&+$value')
            os.environ['LIVE_API_KEY'] = 'environment-key'
            args = parse_args(['--config', str(config), '--api-key', 'cli-key'])
            self.assertEqual(args.server, 'http://example.test:8000')
            self.assertEqual(args.api_key, 'cli-key')
            self.assertEqual(load_settings(config)['LIVE_API_KEY'], 'environment-key')

    def test_urls(self):
        self.assertEqual(publish_url('https://example.test/live/'), 'wss://example.test/live/ws/publish')
        self.assertEqual(publish_url('http://example.test/ws/publish'), 'ws://example.test/ws/publish')
        self.assertEqual(publish_url('localhost:8000'), 'ws://localhost:8000/ws/publish')
        for url in ('ftp://example.test', 'https://user:secret@example.test', 'https://example.test?key=secret'):
            with self.assertRaises(ValueError):
                publish_url(url)

    def test_missing_config_and_partial_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                load_settings(Path(directory) / 'missing.env')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(['--server', 'http://example.test', '--api-key', ''])
            self.assertEqual(error.exception.code, 2)

    def test_linux_without_display(self):
        with patch.dict(os.environ, {}, clear=True), patch('sys.platform', 'linux'):
            self.assertTrue(parse_args([]).headless)


SERVER_DIR = Path(__file__).resolve().parents[1] / 'live-server'


@unittest.skipUnless((SERVER_DIR / 'server.py').is_file(),
                     'dossier live-server/ voisin absent (depot translate seul)')
class LiveServerTests(unittest.TestCase):
    def test_server_model_translation_and_cache(self):
        server_path = SERVER_DIR / 'server.py'
        spec = importlib.util.spec_from_file_location('live_server_model_test', server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.received = (text, kwargs)
                return {}

            def convert_tokens_to_ids(self, token):
                self.token = token
                return 42

            def batch_decode(self, generated, **kwargs):
                return ['A better server translation.']

        class Model:
            def eval(self):
                return self

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[1]]

        service = module.ServerTranslator('test-model')
        service._tokenizer = Tokenizer()
        service._model = Model()
        self.assertEqual(service.translate('Bonjour à tous.', 'Hello everyone.'),
                         'A better server translation.')
        self.assertEqual(service._tokenizer.token, 'eng_Latn')
        self.assertEqual(service._model.kwargs['forced_bos_token_id'], 42)
        self.assertEqual(service.translate('Bonjour à tous.', 'changed fallback'),
                         'A better server translation.')

    def test_authenticated_publisher_viewer_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            history = Path(directory) / 'history.json'
            key = 'integration-secret&+$value'
            env = dict(os.environ, LIVE_API_KEY=key, LIVE_VIEW_CODE='test-view',
                       LIVE_DATA_FILE=str(history), LIVE_TRANSLATION_MODEL='disabled')
            server = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', str(port), '--log-level', 'error'],
                cwd=SERVER_DIR, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            stop = threading.Event()
            publisher = None
            try:
                base = f'http://127.0.0.1:{port}'
                for _ in range(100):
                    try:
                        with urlopen(base + '/health', timeout=0.2):
                            break
                    except OSError:
                        if server.poll() is not None:
                            self.fail(server.stderr.read().decode())
                        time.sleep(0.05)
                else:
                    self.fail('Live Server startup timed out')
                with self.assertRaises(InvalidStatus):
                    with connect(publish_url(base), additional_headers={'x-api-key': 'wrong'}):
                        pass
                with connect(f'ws://127.0.0.1:{port}/ws/watch?code=test-view') as viewer:
                    self.assertEqual(json.loads(viewer.recv(timeout=3))['type'], 'history')
                    publisher = RemotePublisher(base, key, stop)
                    output = Tee(None, publisher)
                    output.put(('interim', 1, 'Hello', 'Bonjour'))
                    self.assertEqual(json.loads(viewer.recv(timeout=5)), {'type': 'interim', 'text': 'Hello'})
                    output.put(('final', 1, 'Hello Linux!', 'Bonjour Linux !'))
                    final = json.loads(viewer.recv(timeout=5))
                    self.assertEqual(final['item']['text'], 'Hello Linux!')
                    output.put(('final', 2, ''))
                    self.assertEqual(json.loads(viewer.recv(timeout=5))['type'], 'live_clear')
                    self.assertEqual(json.loads(history.read_text('utf-8'))['history'][0]['text'], 'Hello Linux!')
            finally:
                stop.set()
                if publisher:
                    publisher.thread.join(timeout=7)
                server.terminate()
                server.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
