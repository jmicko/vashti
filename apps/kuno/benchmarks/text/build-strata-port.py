#!/usr/bin/env python3
"""Build one Rust port artifact, without running incremental numerical gates."""
import hashlib,json,os,pathlib,shutil,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[4]
out=pathlib.Path('/media/jmicko/faststore/kuno-development')/sys.argv[1]
out.mkdir()
options=set(sys.argv[2:])
if options-{'--eight','--resident-224','--resident-240','--intel-core','--gpu-adaptive-usage','--capped-check','--cubek-gemm','--cubek-tilewise','--cubek-m64','--cubek-balanced','--cubek-wide','--native-wide','--native-wide32','--native-expert-wide','--qsa-paired','--qsa-pv-gather','--qsa-local-fault','--native-expert-bulk','--miss-ring','--parallel-loading','--push-descriptors','--gdn-subgroup16','--qsa-fma','--qsa-uniform','--expert-scratch-ring','--qsa-compact','--public-host-clock','--qsa-xmx','--subgroup-publish','--direct-gu','--unused-descriptors','--route-registers','--copy-cohorts','--qsa-packed-cache','--expert-medium','--qsa-reference-schedule','--dequant-record32','--narrow-global','--compute-main','--expert-cohorts','--native-prepared','--qsa-prefetch','--expert-register-prefetch','--qsa-native-xor','--qsa-scale-finite','--subgroup-quad','--expert-macro64','--subgroup-quad-vertical','--qsa-query-planes','--gdn-coalesce-faults','--qsa-head-scatter','--qsa-head-frontier','--macro64-static-geometry','--gr-wide-tail','--tiny-output','--dense-direct-store','--dense-slm40','--dense-prefix','--wide-register-stage','--gr-register-norm','--hc-down-inject','--raw-lookahead','--deep-matrix','--raw-slot-overlap','--dequant-quad32','--expert-ordered-output','--wide-shared-reuse','--dequant-direct8','--small-direct-publish','--small-quantized-matrix','--global-image-experts','--global-register-n64','--global-transpose','--global-wide32'}:raise ValueError('build feature selector')
if '--global-image-experts' in options and not {'--deep-matrix','--small-direct-publish','--expert-cohorts','--wide-shared-reuse'} <= options:raise ValueError('global image experts require complete original owner/publication/cohort bundle')
if '--global-wide32' in options and '--global-transpose' not in options:raise ValueError('wide32 requires complete transpose bundle')
if '--global-transpose' in options and '--global-register-n64' not in options:raise ValueError('transpose requires complete N64/image bundle')
if '--global-register-n64' in options and '--global-image-experts' not in options:raise ValueError('N64 requires complete global image expert bundle')
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
if options & {'--qsa-pv-gather','--qsa-local-fault','--native-expert-bulk','--miss-ring','--parallel-loading','--push-descriptors','--gdn-subgroup16','--qsa-fma','--qsa-uniform','--expert-scratch-ring','--qsa-compact','--public-host-clock','--qsa-xmx'} and '--intel-core' not in options:raise ValueError('coherent native candidates require explicit Intel core')
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
if '--expert-medium' in options and '--direct-gu' not in options:raise ValueError('medium experts require complete direct GU, original tail, and native32 bundle')
features += ['strata-native-expert-medium'] if '--expert-medium' in options else []
if '--subgroup-publish' in options and '--native-expert-wide' not in options:raise ValueError('subgroup publication requires complete native dense and expert wide bundle')
features += ['strata-native-subgroup-publish'] if '--subgroup-publish' in options else []
if '--direct-gu' in options and not {'--subgroup-publish','--native-expert-bulk','--expert-scratch-ring'} <= options:raise ValueError('direct GU requires complete publication, bulk and role scratch bundle')
features += ['strata-native-direct-gu'] if '--direct-gu' in options else []
if '--unused-descriptors' in options and '--direct-gu' not in options:raise ValueError('unused descriptor proof requires complete direct GU bundle')
features += ['strata-native-unused-descriptors'] if '--unused-descriptors' in options else []
if '--route-registers' in options and '--intel-core' not in options:raise ValueError('register router requires Intel core')
features += ['strata-route-registers'] if '--route-registers' in options else []
if '--copy-cohorts' in options and not {'--native-expert-bulk','--miss-ring','--public-host-clock'} <= options:raise ValueError('copy cohorts require complete bulk, miss ring and public clock bundle')
if '--dequant-record32' in options and '--intel-core' not in options:raise ValueError('record32 dequant requires Intel core')
if '--narrow-global' in options and not {'--intel-core','--expert-scratch-ring','--native-expert-bulk','--native-expert-wide'} <= options:raise ValueError('global narrow requires complete padded producer, native matrix and bulk owner bundle')
if '--narrow-global' in options and any(x.startswith('--cubek-') for x in options):raise ValueError('global narrow is a separate native matrix candidate')
if '--compute-main' in options and '--intel-core' not in options:raise ValueError('compute primary requires explicit Intel core and authentic independent queues')
if '--native-prepared' in options and '--intel-core' not in options:raise ValueError('native prepared pipeline requires explicit Intel core and owning graph')
if '--expert-cohorts' in options and not {'--narrow-global','--dequant-record32','--native-expert-bulk','--miss-ring'} <= options:raise ValueError('expert cohorts require complete narrow producer, record32 decoder, native batch and raw ring')
features += ['strata-expert-cohorts'] if '--expert-cohorts' in options else []
if '--expert-register-prefetch' in options and not {'--expert-cohorts','--narrow-global'} <= options:raise ValueError('register prefetch requires complete original narrow/cohort bundle')
features += ['strata-expert-register-prefetch'] if '--expert-register-prefetch' in options else []

features += ['strata-global-register-n64'] if '--global-register-n64' in options else []
features += ['strata-global-transpose'] if '--global-transpose' in options else []
features += ['strata-global-wide32'] if '--global-wide32' in options else []
features += ['strata-global-image-experts'] if '--global-image-experts' in options else []
features += ['strata-native-prepared'] if '--native-prepared' in options else []
features += ['strata-compute-main'] if '--compute-main' in options else []
features += ['strata-narrow-global'] if '--narrow-global' in options else []
features += ['strata-dequant-record32'] if '--dequant-record32' in options else []
features += ['strata-copy-cohorts'] if '--copy-cohorts' in options else []
features += ['strata-qsa-paired'] if '--qsa-paired' in options else []
features += ['strata-qsa-pv-gather'] if '--qsa-pv-gather' in options else []
features += ['strata-qsa-local-fault'] if '--qsa-local-fault' in options else []
features += ['strata-native-expert-bulk'] if '--native-expert-bulk' in options else []
features += ['strata-prefill-miss-ring'] if '--miss-ring' in options else []
features += ['strata-parallel-loading'] if '--parallel-loading' in options else []
features += ['strata-native-push-descriptors'] if '--push-descriptors' in options else []
features += ['strata-gdn-subgroup16'] if '--gdn-subgroup16' in options else []
features += ['strata-qsa-fma'] if '--qsa-fma' in options else []
features += ['strata-qsa-uniform'] if '--qsa-uniform' in options else []
if '--qsa-compact' in options and '--qsa-uniform' not in options:raise ValueError('compact QK requires explicit complete uniform/FMA/paired/gather/fault bundle')
features += ['strata-qsa-compact'] if '--qsa-compact' in options else []
if '--qsa-prefetch' in options and not {'--intel-core','--qsa-paired','--qsa-pv-gather','--qsa-local-fault','--qsa-fma','--qsa-uniform','--qsa-compact'} <= options:raise ValueError('QSA prefetch requires explicit complete paired/gather/fault/FMA/uniform/compact bundle')
if '--qsa-prefetch' in options and options & {'--qsa-packed-cache','--qsa-reference-schedule','--qsa-xmx'}:raise ValueError('QSA prefetch requires the original selected compact scalar source')
features += ['strata-qsa-prefetch'] if '--qsa-prefetch' in options else []
if '--qsa-native-xor' in options and '--qsa-prefetch' not in options:raise ValueError('native XOR requires complete explicit QSA prefetch bundle')
features += ['strata-qsa-native-xor'] if '--qsa-native-xor' in options else []
if '--qsa-scale-finite' in options and '--qsa-native-xor' not in options:raise ValueError('bounded scale predicate requires explicit complete XOR/prefetch bundle')
features += ['strata-qsa-scale-finite'] if '--qsa-scale-finite' in options else []
if '--subgroup-quad' in options and '--qsa-scale-finite' not in options:raise ValueError('native quad requires explicit complete scale/XOR/prefetch bundle')
features += ['strata-subgroup-quad'] if '--subgroup-quad' in options else []
if '--subgroup-quad-vertical' in options and '--subgroup-quad' not in options:raise ValueError('vertical native quad requires explicit complete quad bundle')
features += ['strata-subgroup-quad-vertical'] if '--subgroup-quad-vertical' in options else []
if '--expert-macro64' in options and '--subgroup-quad-vertical' not in options:raise ValueError('expert macro64 requires explicit complete native quad bundle')
features += ['strata-expert-macro64'] if '--expert-macro64' in options else []
if '--qsa-query-planes' in options and '--expert-macro64' not in options:raise ValueError('query planes require explicit complete expert/vertical quad bundle')
features += ['strata-qsa-query-planes'] if '--qsa-query-planes' in options else []
if '--gdn-coalesce-faults' in options and '--qsa-query-planes' not in options:raise ValueError('GDN fault coalescing requires explicit complete query plane bundle')
features += ['strata-gdn-coalesce-faults'] if '--gdn-coalesce-faults' in options else []
if '--qsa-head-scatter' in options and '--gdn-coalesce-faults' not in options:raise ValueError('ordered head scatter requires explicit complete query/GDN bundle')
features += ['strata-qsa-head-scatter'] if '--qsa-head-scatter' in options else []
if '--qsa-head-frontier' in options and '--qsa-head-scatter' not in options:raise ValueError('ordered frontier requires explicit complete head-scatter bundle')
features += ['strata-qsa-head-frontier'] if '--qsa-head-frontier' in options else []
if '--macro64-static-geometry' in options and '--qsa-head-frontier' not in options:raise ValueError('static expert geometry requires explicit complete ordered-frontier bundle')
features += ['strata-macro64-static-geometry'] if '--macro64-static-geometry' in options else []
if '--gr-wide-tail' in options and not {'--macro64-static-geometry','--native-wide32'} <= options:raise ValueError('GR wide tail requires complete static-geometry and native32 bundle')
features += ['strata-gr-wide-tail'] if '--gr-wide-tail' in options else []
if '--tiny-output' in options and '--gr-wide-tail' not in options:raise ValueError('tiny output requires the complete GR/native bundle')
features += ['strata-tiny-output'] if '--tiny-output' in options else []
if '--dense-direct-store' in options and '--tiny-output' not in options:raise ValueError('direct dense publication requires the complete tiny-output bundle')
features += ['strata-dense-direct-store'] if '--dense-direct-store' in options else []
if '--dense-slm40' in options and '--dense-direct-store' not in options:raise ValueError('padded dense operands require the complete direct publication bundle')
features += ['strata-dense-slm40'] if '--dense-slm40' in options else []
if '--dense-prefix' in options and '--dense-slm40' not in options:raise ValueError('wide prefix requires complete padded/direct bundle')
features += ['strata-dense-prefix'] if '--dense-prefix' in options else []
if '--wide-register-stage' in options and '--dense-prefix' not in options:raise ValueError('wide register staging requires complete prefix/padded/direct bundle')
features += ['strata-wide-register-stage'] if '--wide-register-stage' in options else []
if '--gr-register-norm' in options and '--wide-register-stage' not in options:raise ValueError('GR register normalization requires complete coupled pipeline bundle')
features += ['strata-gr-register-norm'] if '--gr-register-norm' in options else []
if '--hc-down-inject' in options and not {'--gr-register-norm','--subgroup-publish','--native-wide32'} <= options:raise ValueError('HC down/injection fusion requires the complete register-normalized native subgroup bundle')
features += ['strata-hc-down-inject'] if '--hc-down-inject' in options else []
if '--raw-lookahead' in options and not {'--gr-register-norm','--expert-cohorts','--copy-cohorts','--miss-ring'} <= options:raise ValueError('raw lookahead requires the complete normalized two-bank cohort bundle')
features += ['strata-raw-lookahead'] if '--raw-lookahead' in options else []
if '--deep-matrix' in options and not {'--hc-down-inject','--native-expert-wide','--expert-macro64'} <= options:raise ValueError('deep matrix requires the complete HC/native expert bundle')
features += ['strata-deep-matrix'] if '--deep-matrix' in options else []
if '--raw-slot-overlap' in options and '--raw-lookahead' not in options:raise ValueError('per-slot overlap requires the complete eight-expert lookahead bundle')
features += ['strata-raw-slot-overlap'] if '--raw-slot-overlap' in options else []
if '--dequant-quad32' in options and not {'--dequant-record32','--intel-core'} <= options:raise ValueError('coalesced record32 requires complete original record32/Intel controls')
features += ['strata-dequant-quad32'] if '--dequant-quad32' in options else []
if '--dequant-direct8' in options and '--dequant-quad32' not in options:raise ValueError('direct8 requires original complete record32/quad controls')
features += ['strata-dequant-direct8'] if '--dequant-direct8' in options else []
if '--expert-ordered-output' in options and not {'--direct-gu','--expert-cohorts'} <= options:raise ValueError('expert-ordered output requires the complete native expert bundle')
features += ['strata-expert-ordered-output'] if '--expert-ordered-output' in options else []
if '--wide-shared-reuse' in options and not {'--expert-ordered-output','--subgroup-publish','--native-expert-wide'} <= options:raise ValueError('shared lifetime reuse requires the complete ordered/native subgroup bundle')
features += ['strata-wide-shared-reuse'] if '--wide-shared-reuse' in options else []
if '--small-direct-publish' in options and not {'--expert-ordered-output','--subgroup-publish','--direct-gu','--expert-cohorts'} <= options:raise ValueError('small expert publication requires the complete subgroup, raw GU, ordered DOWN and cohort bundle')
features += ['strata-small-direct-publish'] if '--small-direct-publish' in options else []
if '--small-quantized-matrix' in options and not {'--small-direct-publish','--intel-core','--expert-cohorts'} <= options:raise ValueError('small quantized matrix requires the complete small publication, Intel format and cohort bundle')
features += ['strata-small-quantized-matrix'] if '--small-quantized-matrix' in options else []


if {'--qsa-packed-cache','--qsa-xmx'} <= options:raise ValueError('packed-cache and matrix QSA are separate complete candidates')
if '--qsa-packed-cache' in options and '--qsa-compact' not in options:raise ValueError('packed QK cache requires complete compact attention bundle')
features += ['strata-qsa-packed-cache'] if '--qsa-packed-cache' in options else []
if '--qsa-reference-schedule' in options and '--qsa-compact' not in options:raise ValueError('reference schedule requires explicit existing complete attention bundle')
features += ['strata-qsa-reference-schedule'] if '--qsa-reference-schedule' in options else []
if '--qsa-xmx' in options and '--qsa-compact' not in options:raise ValueError('matrix attention requires explicit complete compact attention bundle')
features += ['strata-qsa-xmx'] if '--qsa-xmx' in options else []
features += ['strata-public-host-clock'] if '--public-host-clock' in options else []
if '--expert-scratch-ring' in options and not {'--intel-core','--native-expert-bulk','--miss-ring'} <= options:raise ValueError('scratch ring requires explicit complete native bulk and miss ring')
features += ['strata-expert-scratch-ring'] if '--expert-scratch-ring' in options else []
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
