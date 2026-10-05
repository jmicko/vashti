#!/usr/bin/env python3
"""Regenerate deterministic source diff/provenance from pinned cached archives."""
from pathlib import Path
import difflib
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
lab = root.parent
cache = Path('/home/jmicko/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f')
manifest = {
    'scope': 'standalone native-text laboratory ownership sources; acceptance is recorded separately',
    'upstream_git_commit': '7cf203735e095e640a2c03b2400d0faa03196bb4',
    'crates': {},
    'unchanged_wgpu_version': '29.0.4',
    'changed_files': [],
}
patch = []
for crate in ('cubecl-runtime', 'cubecl-core', 'cubecl-wgpu'):
    name = crate + '-0.10.0'
    archive = cache / (name + '.crate')
    with tarfile.open(archive) as tar:
        originals = {
            member.name[len(name) + 1:]: tar.extractfile(member).read()
            for member in tar.getmembers() if member.isfile()
        }
    vcs = json.loads(originals['.cargo_vcs_info.json'])
    assert vcs['git']['sha1'] == manifest['upstream_git_commit']
    manifest['crates'][crate] = {
        'version': '0.10.0',
        'published_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'git': vcs,
        'licenses': ['MIT', 'Apache-2.0'],
    }
    vendor = lab / 'vendor' / crate
    for path in sorted(vendor.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(vendor))
        modified = path.read_bytes()
        original = originals.get(relative, b'')
        if modified == original:
            continue
        # Cargo unpack markers are local artifacts, not vendored source changes.
        if relative in ('.cargo-ok', '.cargo-checksum.json'):
            continue
        key = crate + '/' + relative
        manifest['changed_files'].append({
            'path': key,
            'sha256': hashlib.sha256(modified).hexdigest(),
            'original_sha256': hashlib.sha256(original).hexdigest() if relative in originals else None,
        })
        patch.extend(difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            modified.decode().splitlines(keepends=True),
            fromfile='a/' + key if relative in originals else '/dev/null',
            tofile='b/' + key,
        ))
    for license_name in ('LICENSE-MIT', 'LICENSE-APACHE'):
        assert (vendor / license_name).read_bytes() == originals[license_name]
text = ''.join(patch)
(root / 'transient-runtime.patch').write_text(text)
manifest['patch_sha256'] = hashlib.sha256(text.encode()).hexdigest()
manifest['lock_sha256'] = hashlib.sha256((lab / 'Cargo.lock').read_bytes()).hexdigest()
manifest['host_lock_sha256'] = hashlib.sha256((root / 'host/Cargo.lock').read_bytes()).hexdigest()
(root / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(len(manifest['changed_files']), 'modified/new source files')
print('patch SHA-256:', manifest['patch_sha256'])
