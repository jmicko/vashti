const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(`${__dirname}/../static/release-ui.js`, 'utf8'), context);
const { compareVersions } = context.window.VashtiReleaseUI;
const versions = [
  'v0.12.0', 'v0.13.0-alpha', 'v0.13.0-alpha.1', 'v0.13.0-beta.1',
  'v0.13.0-beta.2', 'v0.13.0-beta.10', 'v0.13.0-rc.1', 'v0.13.0', 'v0.14.0-beta.1',
];
for (let i = 0; i < versions.length; i++) {
  for (let j = 0; j < versions.length; j++) {
    assert.equal(Math.sign(compareVersions(versions[i], versions[j])), Math.sign(i - j));
  }
}
console.log('SemVer release UI ordering passed');
