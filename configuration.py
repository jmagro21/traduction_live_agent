"""Configuration partageable avec le fichier .env de Docker Compose."""

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values


def app_dir():
    """Dossier du script, ou de l'executable une fois compile (PyInstaller)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


SETTING_NAMES = (
    'LIVE_SERVER', 'LIVE_API_KEY',
    'TRANSLATE_ENGINE', 'TRANSLATE_LANGUAGE', 'TRANSLATE_MIC_DEVICE',
    'TRANSLATE_FONT_SIZE', 'TRANSLATE_THRESHOLD', 'TRANSLATE_HEADLESS',
    'WHISPER_MODEL', 'WHISPER_DEVICE', 'WHISPER_GPU_INDEX', 'WHISPER_BEAM_FINAL',
    'VOSK_MODEL',
)


def load_settings(filename=None):
    path = Path(filename) if filename else app_dir() / '.env'
    if filename and not path.is_file():
        raise ValueError(f"Fichier de configuration introuvable : {path}")
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    return {name: os.environ.get(name, values.get(name))
            for name in SETTING_NAMES}


def publish_url(base):
    base = base.strip()
    if '://' not in base:
        base = 'http://' + base
    parts = urlsplit(base)
    if (parts.scheme not in ('http', 'https', 'ws', 'wss') or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise ValueError('LIVE_SERVER doit etre une URL http(s) ou ws(s), sans identifiants, parametres ni fragment.')
    try:
        parts.port
    except ValueError as exc:
        raise ValueError('Port LIVE_SERVER invalide.') from exc
    path = parts.path.rstrip('/')
    if not path.endswith('/ws/publish'):
        path += '/ws/publish'
    return urlunsplit(({'http': 'ws', 'https': 'wss'}.get(parts.scheme, parts.scheme),
                       parts.netloc, path, '', ''))
