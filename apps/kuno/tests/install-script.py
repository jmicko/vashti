"""Run without sudo or network: python3 apps/kuno/tests/install-script.py."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'packaging/install.sh'

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.archive = self.root / 'kuno-linux-x86_64.tar.gz'
        payload = b'#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_ROOT/invoked"\n'
        with tarfile.open(self.archive, 'w:gz') as archive:
            info = tarfile.TarInfo('kuno')
            info.mode = 0o755
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        (self.root / 'SHA256SUMS').write_text(f'{digest}  {self.archive.name}\n')
        self.command('uname', 'case "$1" in -s) echo Linux;; -m) echo x86_64;; esac')
        self.command('id', 'echo 0')  # Invokes only our fake downloaded executable.
        self.command('curl', '''
url= destination=
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) destination="$2"; shift 2 ;;
    --proto|--proto-redir) shift 2 ;;
    https://*) url="$1"; shift ;;
    *) shift ;;
  esac
done
case "$url" in
  */VERSION) echo v0.13.0-beta.1 ;;
  *) cp "$TEST_ROOT/${url##*/}" "$destination" ;;
esac
''')

    def command(self, name, body):
        path = self.bin / name
        path.write_text('#!/bin/sh\nset -eu\n' + body + '\n')
        path.chmod(0o755)

    def run_installer(self, **overrides):
        env = {**os.environ, 'PATH': f'{self.bin}:/usr/bin:/bin',
               'TEST_ROOT': str(self.root), 'KUNO_VERSION': 'v0.13.0-beta.1',
               'KUNO_YES': '1', **overrides}
        return subprocess.run(['sh', str(SCRIPT)], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, start_new_session=True)

    def test_verified_download_reaches_installer_with_explicit_yes(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root / 'invoked').read_text().splitlines()
        self.assertIn(args[0], ['install', 'upgrade'])
        self.assertEqual(args[1:], ['--yes'])

    def test_tampered_download_never_executes(self):
        self.archive.write_bytes(b'bad download')
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse((self.root / 'invoked').exists())

    def test_piped_noninteractive_install_needs_explicit_consent(self):
        result = self.run_installer(KUNO_YES='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No terminal', result.stderr)
        self.assertFalse((self.root / 'invoked').exists())

    def test_unsafe_version_never_executes(self):
        self.assertNotEqual(self.run_installer(KUNO_VERSION='v1/../../etc').returncode, 0)
        self.assertFalse((self.root / 'invoked').exists())

if __name__ == '__main__':
    unittest.main()
