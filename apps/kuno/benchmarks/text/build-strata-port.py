#!/usr/bin/env python3
"""Build one Rust port artifact, without running incremental numerical gates."""
import hashlib,json,os,pathlib,shutil,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[4]
out=pathlib.Path('/media/jmicko/faststore/kuno-development')/sys.argv[1]
out.mkdir()
options=set(sys.argv[2:])
if options-{'--eight','--resident-224','--resident-240','--intel-core','--gpu-adaptive-usage','--capped-check'}:raise ValueError('build feature selector')
if '--gpu-adaptive-usage' in options and '--intel-core' not in options:raise ValueError('GPU route usage requires explicit Intel core selection')
if '--capped-check' in options and '--intel-core' not in options:raise ValueError('fixed public cap requires explicit Intel core')
features=['strata-rust-port']+(['strata-eight-window'] if '--eight' in options else [])+(['strata-resident-224'] if '--resident-224' in options else [])+(['strata-resident-240'] if '--resident-240' in options else [])+(['strata-intel-core'] if '--intel-core' in options else [])+(['strata-gpu-adaptive-usage'] if '--gpu-adaptive-usage' in options else [])
features += ['strata-public-cap-check'] if '--capped-check' in options else []
source_paths=['apps/kuno/benchmarks/native-text','crates/kuno-text','apps/kuno/src/native_protocol.rs','apps/kuno/src/image_privacy.rs','apps/kuno/benchmarks/text/build-strata-port.py','apps/kuno/benchmarks/text/run-strata-port.py']
subprocess.run(['git','diff','--quiet','HEAD','--',*source_paths],cwd=root,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
(out/'source-commit.txt').write_text(commit+'\n')
(out/'build-features.json').write_text(json.dumps({'features':features})+'\n')
env=dict(os.environ,CARGO_TARGET_DIR='/media/jmicko/faststore/kuno-development/root-native-text-target-20261005',CARGO_BUILD_JOBS='2')
with (out/'cargo-artifacts.jsonl').open('w') as stdout,(out/'cargo-build.log').open('w') as stderr:
 subprocess.run(['cargo','build','--locked','--release','--manifest-path','apps/kuno/benchmarks/native-text/Cargo.toml','--bin','kuno-strata-rust','--features',','.join(features),'--message-format=json'],cwd=root,env=env,stdout=stdout,stderr=stderr,check=True)
subprocess.run(['git','diff','--quiet','HEAD','--',*source_paths],cwd=root,check=True)
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=commit:raise ValueError('source changed during build')
records=[json.loads(x) for x in (out/'cargo-artifacts.jsonl').read_text().splitlines()]
paths=[x['executable'] for x in records if x.get('reason')=='compiler-artifact' and x['target']['name']=='kuno-strata-rust' and x.get('executable')]
assert len(paths)==1
source=pathlib.Path(paths[0]);digest=hashlib.sha256(source.read_bytes()).hexdigest()
frozen=out/('kuno-strata-rust-'+digest);shutil.copyfile(source,frozen);frozen.chmod(0o500)
print(json.dumps({'source_commit':commit,'frozen_sha256':digest,'artifact':str(frozen)}))
