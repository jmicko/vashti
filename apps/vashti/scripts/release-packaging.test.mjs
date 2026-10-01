import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const source = join(dirname(fileURLToPath(import.meta.url)), 'publish-release.sh');

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'vashti-release-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const path of ['apps/vashti/scripts', '.private/keys', 'crates/vashti-update-manifest'])
    mkdirSync(join(root, path), { recursive: true });
  copyFileSync(source, join(root, 'apps/vashti/scripts/publish-release.sh'));
  writeFileSync(join(root, 'apps/vashti/Cargo.toml'), 'version = "0.13.0-beta.2"\n');
  const key = join(root, '.private/keys/vashti-release-signing-key.pem');
  const generated = spawnSync('openssl', ['genpkey', '-algorithm', 'ED25519', '-out', key]);
  assert.equal(generated.status, 0);
  const publicKey = spawnSync('openssl', ['pkey', '-in', key, '-pubout', '-outform', 'DER']);
  assert.equal(publicKey.status, 0);
  writeFileSync(join(root, 'crates/vashti-update-manifest/release-public-key.txt'),
    publicKey.stdout.subarray(-32).toString('base64'));
  const release = join(root, 'apps/vashti/dist/release/v0.13.0-beta.2');
  mkdirSync(release, { recursive: true });
  for (const name of ['vashti-linux-x86_64.tar.gz', 'kuno-linux-x86_64.tar.gz'])
    writeFileSync(join(release, name), 'disposable artifact fixture');
  return {
    add(name) { writeFileSync(join(release, name), 'disposable artifact fixture'); },
    run(extra = {}) {
      return spawnSync('bash', [join(root, 'apps/vashti/scripts/publish-release.sh')], {
        cwd: root,
        encoding: 'utf8',
        env: { ...process.env, VERSION: 'v0.13.0-beta.2', ARTIFACT: '', NOTES: '',
          RELEASE_NOTES_FILE: '', VASHTI_RELEASE_SIGNING_KEY: key, CHECK_ONLY: '1',
          REQUIRE_COMPLETE_RELEASE: '1', REQUIRE_KUNO_SYCL_RUNTIME: '0', ...extra },
      });
    },
  };
}

test('versioned APK and optional runtime are signed in one complete release', t => {
  const f = fixture(t);
  f.add('vashti-0.13.0-beta.2.apk');
  f.add('kuno-sycl-linux-x86_64.tar.gz');
  const result = f.run({ REQUIRE_KUNO_SYCL_RUNTIME: '1' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /4 artifacts.*No upload attempted/);
});

test('legacy APK filename remains accepted', t => {
  const f = fixture(t);
  f.add('vashti-android.apk');
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /3 artifacts/);
});

test('only the versioned APK is used when a legacy copy is also present', t => {
  const f = fixture(t);
  f.add('vashti-0.13.0-beta.2.apk');
  f.add('vashti-android.apk');
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /3 artifacts/);
});

test('required missing runtime fails before consuming an upload key', t => {
  const f = fixture(t);
  f.add('vashti-0.13.0-beta.2.apk');
  const result = f.run({ REQUIRE_KUNO_SYCL_RUNTIME: '1' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /missing required release target: kuno-sycl-linux-x86_64/);
});
