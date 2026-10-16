#!/usr/bin/env python3
"""Install the frozen CPU runtime into a dedicated .venv-current environment."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
    raise SystemExit('The frozen runtime installation supports Linux x86-64, Python 3.12.')
record = json.loads((ROOT/'reference/2026-10-15/runtime.json').read_text())
wheel = ROOT/'vendor'/record['wheel']
if not wheel.exists():
    raise SystemExit('Missing frozen wheel. Unpack the complete reproduction bundle (including vendor/).')
if hashlib.sha256(wheel.read_bytes()).hexdigest() != record['wheel_sha256']:
    raise SystemExit('Frozen AISim wheel hash mismatch.')
environment = ROOT/'.venv-current'
venv.EnvBuilder(with_pip=True).create(environment)
python = str(environment/'bin/python')
subprocess.run([python, '-m', 'pip', 'install', '--require-hashes', '-r',
                str(ROOT/'environment/requirements-current.lock')], check=True)
subprocess.run([python, '-m', 'pip', 'install', '--no-deps', str(wheel)], check=True)
subprocess.run([python, str(ROOT/'reproduce.py'), 'doctor'], check=True)
