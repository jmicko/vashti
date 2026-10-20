#!/usr/bin/env python3
"""Build one Rust port artifact, without running incremental numerical gates."""
import hashlib,json,os,pathlib,shutil,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[4]
out=pathlib.Path('/media/jmicko/faststore/kuno-development')/sys.argv[1]
out.mkdir()
options=set(sys.argv[2:])
if options-{'--eight','--resident-224','--resident-240','--intel-core','--gpu-adaptive-usage','--capped-check','--cubek-gemm','--cubek-tilewise','--cubek-m64','--cubek-balanced','--cubek-wide','--native-wide','--native-wide32','--native-expert-wide','--qsa-paired','--qsa-pv-gather','--qsa-local-fault','--native-expert-bulk','--miss-ring','--parallel-loading','--push-descriptors'}:raise ValueError('build feature selector')
if '--gpu-adaptive-usage' in options and '--intel-core' not in options:raise ValueError('GPU route usage requires explicit Intel core selection')
if '--capped-check' in options and '--intel-core' not in options:raise ValueError('fixed public cap requires explicit Intel core')
if '--cubek-gemm' in options and '--intel-core' not in options:raise ValueError('CubeK candidate requires explicit Intel core')
if '--cubek-tilewise' in options and not {'--intel-core','--cubek-gemm'} <= options:raise ValueError('CubeK tilewise candidate requires explicit Intel core and GEMM')
if '--cubek-m64' in options and not {'--intel-core','--cubek-gemm','--cubek-tilewise'} <= options:raise ValueError('CubeK M64 candidate requires explicit complete Tilewise selection')
if '--cubek-balanced' in options and not {'--intel-core','--cubek-gemm','--cubek-tilewise'} <= options:raise ValueError('CubeK balanced candidate requires explicit complete Tilewise selection')
if '--cubek-wide' in options and not {'--intel-core','--cubek-gemm','--cubek-tilewise','--cubek-balanced'} <= options:raise ValueError('CubeK wide candidate requires explicit complete balanced Tilewise selection')
if '--native-wide' in options and ('--intel-core' not in options or any(x.startswith('--cubek-') for x in options)):raise ValueError('native wide candidate requires Intel core and separate CubeK selection')
if '--native-wide32' in options and '--native-wide' not in options:raise ValueError('native 32-subgroup candidate requires explicit full native wide selection')
if '--native-expert-wide' in options and not {'--native-wide32','--native-expert-bulk'} <= options:raise ValueError('routed native wide candidate requires explicit native32 and complete expert bulk selection')
if '--qsa-paired' in options and '--intel-core' not in options:raise ValueError('paired prompt attention requires explicit Intel core')
if '--miss-ring' in options and not {'--intel-core','--native-expert-bulk'} <= options:raise ValueError('miss-only prefill ring requires explicit Intel core and complete expert batch')
if options & {'--qsa-pv-gather','--qsa-local-fault','--native-expert-bulk','--miss-ring','--parallel-loading','--push-descriptors'} and '--intel-core' not in options:raise ValueError('coherent native candidates require explicit Intel core')
features=['strata-rust-port']+(['strata-eight-window'] if '--eight' in options else [])+(['strata-resident-224'] if '--resident-224' in options else [])+(['strata-resident-240'] if '--resident-240' in options else [])+(['strata-intel-core'] if '--intel-core' in options else [])+(['strata-gpu-adaptive-usage'] if '--gpu-adaptive-usage' in options else [])
features += ['strata-public-cap-check'] if '--capped-check' in options else []
features += ['strata-cubek-gemm'] if '--cubek-gemm' in options else []
features += ['strata-cubek-tilewise'] if '--cubek-tilewise' in options else []
features += ['strata-cubek-m64'] if '--cubek-m64' in options else []
features += ['strata-cubek-balanced'] if '--cubek-balanced' in options else []
features += ['strata-cubek-wide'] if '--cubek-wide' in options else []
features += ['strata-native-wide'] if '--native-wide' in options else []
features += ['strata-native-wide32'] if '--native-wide32' in options else []
features += ['strata-native-expert-wide'] if '--native-expert-wide' in options else []
features += ['strata-qsa-paired'] if '--qsa-paired' in options else []
features += ['strata-qsa-pv-gather'] if '--qsa-pv-gather' in options else []
features += ['strata-qsa-local-fault'] if '--qsa-local-fault' in options else []
features += ['strata-native-expert-bulk'] if '--native-expert-bulk' in options else []
features += ['strata-prefill-miss-ring'] if '--miss-ring' in options else []
features += ['strata-parallel-loading'] if '--parallel-loading' in options else []
features += ['strata-native-push-descriptors'] if '--push-descriptors' in options else []
source_paths=['apps/kuno/benchmarks/native-text','crates/kuno-text','apps/kuno/src/native_protocol.rs','apps/kuno/src/native_artifact.rs','apps/kuno/src/image_privacy.rs','apps/kuno/benchmarks/text/build-strata-port.py','apps/kuno/benchmarks/text/run-strata-port.py']
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
