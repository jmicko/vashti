#!/usr/bin/env python3
"""CPU-only managed-builder source/environment fixtures; no build tools or GPU."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
path = Path(__file__).with_name('build-image-runtime-sycl.py')
spec = importlib.util.spec_from_file_location('managed_builder_under_test', path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class SourceFixtures(unittest.TestCase):
    def setUp(self):
        parent = builder.ROOT / 'target/managed-sycl-builder-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.source = Path(self.temp.name)
        for root in [self.source, self.source / 'ggml']:
            root.mkdir(exist_ok=True)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('ignored.cpp\n')
            subprocess.run(['git', '-C', str(root), 'add', '.gitignore'], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_pristine_and_intent_to_add_are_allowed(self):
        patch_file = self.source / 'ggml/reviewed.cpp'
        patch_file.write_text('// synthetic reviewed addition\n')
        subprocess.run(['git', '-C', str(patch_file.parent), 'add', '--intent-to-add', patch_file.name], check=True)
        builder.reject_untracked_source(self.source)

    def test_untracked_engine_file_is_rejected(self):
        (self.source / 'unknown.cpp').write_text('// synthetic\n')
        with self.assertRaisesRegex(ValueError, 'engine source contains'):
            builder.reject_untracked_source(self.source)

    def test_untracked_GGML_glob_file_is_rejected(self):
        (self.source / 'ggml/unknown.cpp').write_text('// synthetic\n')
        with self.assertRaisesRegex(ValueError, 'GGML source contains'):
            builder.reject_untracked_source(self.source)

    def test_ignored_engine_file_is_rejected(self):
        (self.source / 'ignored.cpp').write_text('// synthetic ignored\n')
        with self.assertRaisesRegex(ValueError, 'engine source contains'):
            builder.reject_untracked_source(self.source)

    def test_ignored_GGML_glob_file_is_rejected(self):
        (self.source / 'ggml/ignored.cpp').write_text('// synthetic ignored\n')
        with self.assertRaisesRegex(ValueError, 'GGML source contains'):
            builder.reject_untracked_source(self.source)

    def test_encoded_and_unreviewed_environment_cannot_override_linker(self):
        original = os.environ.copy()
        try:
            for key in builder.CLEARED_BUILD_ENV:
                os.environ[key] = 'synthetic_unreviewed_flag'
            os.environ['RUSTFLAGS'] = 'synthetic_unreviewed_linker'
            result = builder.build_environment({'RUSTFLAGS': '-Clinker=/synthetic/reviewed-linker'})
            self.assertTrue(all(key not in result for key in builder.CLEARED_BUILD_ENV))
            self.assertEqual(result['RUSTFLAGS'], '-Clinker=/synthetic/reviewed-linker')
            self.assertEqual(result['PATH'], original['PATH'])
        finally:
            os.environ.clear()
            os.environ.update(original)


if __name__ == '__main__':
    unittest.main()
