"""Test actual signed packages against a temporary local Hub, never the live site.
Requires package-hub.sh, all release packages, and the release signing key.
Run from the repository root: python3 apps/vashti-hub/tests/release-upload-smoke.py
"""
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
version = 'v' + re.search(r'^version = "([^"]+)"', (ROOT / 'apps/vashti/Cargo.toml').read_text(), re.M)[1]
base = 'http://127.0.0.1:17883'
with tempfile.TemporaryDirectory(prefix='vashti-upload-smoke-') as temporary:
    directory = Path(temporary)
    server = subprocess.Popen(
        [str(ROOT / 'apps/vashti-hub/dist/hub/vashti-hub')],
        env={**os.environ, 'VASHTI_HUB_DATA_DIR': temporary, 'VASHTI_HUB_BIND': '127.0.0.1:17883'},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(base + path, data=body,
                                     headers={'Content-Type': 'application/json', 'Origin': base})
        return opener.open(req, timeout=15).read()

    def rss_kib():
        status = Path(f'/proc/{server.pid}/status').read_text()
        return int(re.search(r'^VmRSS:\s+(\d+)', status, re.M)[1])

    publisher = None
    try:
        for _ in range(100):
            try:
                request('/api/admin/status')
                break
            except OSError:
                if server.poll() is not None:
                    raise RuntimeError('Local Hub exited')
                time.sleep(.1)
        else:
            raise RuntimeError('Local Hub startup timed out')
        request('/api/admin/setup', {
            'setup_key': (directory / 'admin-setup-key.txt').read_text().strip(),
            'password': secrets.token_urlsafe(32),
        })
        # Match production's existing stable release without copying any real database.
        with sqlite3.connect(directory / 'hub.db') as db:
            db.execute("INSERT INTO releases(version,version_major,version_minor,version_patch,release_status,created_at) VALUES('v0.12.0',0,12,0,'released',1)")
            db.execute("UPDATE release_settings SET latest_version='v0.12.0' WHERE id=1")
        key = json.loads(request('/api/admin/upload-key', {}))['token']
        publisher = subprocess.Popen(
            [str(ROOT / 'apps/vashti/scripts/publish-release.sh')], cwd=ROOT,
            env={**os.environ, 'VASHTI_HUB_URL': base, 'VASHTI_HUB_UPLOAD_KEY': key, 'REQUIRE_COMPLETE_RELEASE': '1'},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        del key
        peak = rss_kib()
        deadline = time.monotonic() + 180
        while publisher.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Local upload test timed out')
            peak = max(peak, rss_kib())
            time.sleep(.05)
        output, error = publisher.communicate()
        if publisher.returncode:
            raise RuntimeError(f'Local publisher failed: {error}')
        with sqlite3.connect(directory / 'hub.db') as db:
            rows = db.execute('SELECT filename,sha256,size_bytes,storage_path FROM release_artifacts WHERE release_version=?', (version,)).fetchall()
            assert len(rows) == 3
            for filename, digest, size, storage in rows:
                with open(storage, 'rb') as file:
                    assert hashlib.file_digest(file, 'sha256').hexdigest() == digest, filename
                assert Path(storage).stat().st_size == size
        assert request('/releases/latest/VERSION').strip() == b'v0.12.0'
        manifest = json.loads(request('/api/updates/prerelease/kuno-linux-x86_64'))
        assert manifest['artifact']['version'] == version
        assert not list((directory / 'artifacts').glob('.upload-*'))
        assert peak < 96 * 1024, f'Upload memory usage too high: {peak} KiB'
        print(f'Signed three-artifact upload passed; peak Hub RSS {peak / 1024:.1f} MiB; stable unchanged; temporary files cleaned.')
    finally:
        if publisher is not None and publisher.poll() is None:
            publisher.kill()
            publisher.wait()
        server.terminate()
        server.wait(timeout=10)
