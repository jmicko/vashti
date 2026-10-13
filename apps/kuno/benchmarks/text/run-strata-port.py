#!/usr/bin/env python3
"""Fixed PUBLIC fixtures only. Inference and request history remain in Rust/RAM."""
import datetime,hashlib,importlib.util,json,os,pathlib,pwd,subprocess,sys,time
BASE=pathlib.Path('/media/jmicko/faststore/kuno-development')
ROOT=pathlib.Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('protected',BASE/'run-native-strata-public-protected-20261009.py')
protected=importlib.util.module_from_spec(spec);spec.loader.exec_module(protected)

def helper(name,*args):
 return json.loads(subprocess.check_output(['python3',str(BASE/'helpers-20261008'/name),*args],text=True))

def run():
 if os.getuid()!=0 or len(sys.argv)!=5:raise ValueError('root BUILD OUT code|prose q8|q2')
 build=BASE/sys.argv[1];out=BASE/sys.argv[2];fixture,profile=sys.argv[3:]
 if out.exists():raise ValueError('fresh output required')
 out.mkdir()
 account=pwd.getpwnam('jmicko');os.chown(out,account.pw_uid,account.pw_gid)
 bins=list(build.glob('kuno-strata-rust-*'))
 if len(bins)!=1:raise ValueError('one immutable Rust artifact required')
 binary=bins[0];digest=hashlib.sha256(binary.read_bytes()).hexdigest()
 if binary.name!='kuno-strata-rust-'+digest:raise ValueError('artifact hash')
 if fixture not in ('code','prose') or profile not in ('q8','q2'):raise ValueError('public scope')
 label=f'language-{fixture}-intel-mtp-four-lookahead-down-integer-grouped-ple-gu-stride-shared-head-gpu-host'
 env,memory,mode=protected.select_trial(label,out/'intel.json')
 if profile=='q2':
  env['KUNO_TEST_MTP']=str(BASE/'kuno-strata-mtp-q2-20261009/mtp-q2_0.gguf')
  env['KUNO_LAB_MTP_PROFILE']='strata-q2';env.pop('KUNO_LAB_MTP_Q8_HEAD',None)
 env.pop('KUNO_LAB_MTP_REPAIR',None)
 # Capture source identity; whole files were hashed by preceding accepted trials.
 sources=[]
 for path in sorted(protected.MODEL.glob('*.gguf')):
  st=path.stat();sources.append({'path':str(path),'size':st.st_size,'mtime_ns':st.st_mtime_ns})
 (out/'source.json').write_text(json.dumps({'commit':(build/'source-commit.txt').read_text().strip(),'artifact_sha256':digest,'features':['strata-rust-port'],'fixture':fixture,'draft_profile':profile,'seed':424242,'temperature':0,'sources':sources,'source_full_sha_recheck_this_run':False,'original_full_sha_evidence':str(BASE/'native-next-v197-staged-down-code-intel-20261013/public-original-source-verification.json')},indent=2)+'\n')
 before=helper('final-secondary-state.py');(out/'host-before.json').write_text(json.dumps(before,indent=2)+'\n')
 if before['nvidia_compute_process_count'] or any(r['used_bytes']>160*1024**2 for r in before['intel']['xe_memory']):raise ValueError('GPU not free')
 if before['admission']['state']['admission']['active'] or before['admission']['state']['admission']['waiting']:raise ValueError('admission not idle')
 try:
  helper('host-state.py','pause')
  snapshot=protected.intel_mtp_snapshot()
  env.update({'KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES':str(snapshot['total_bytes']),'KUNO_PUBLIC_MTP_INTEL_USED_BYTES':str(snapshot['used_bytes']),'KUNO_PUBLIC_MTP_INTEL_DEVICE_ID':str(snapshot['device_id'])})
  (out/'intel-mtp-memory-before-models.json').write_text(json.dumps(snapshot,indent=2)+'\n')
  # Remove only laboratory selectors from inherited environment. Secrets are never printed.
  child={k:v for k,v in os.environ.items() if not k.startswith(('KUNO_LAB_','KUNO_TEST_','KUNO_PUBLIC_'))}
  child.update(env)
  cmd=['systemd-run','--quiet','--wait','--pipe','--collect','--unit='+out.name,'--uid=jmicko','--property=MemoryMax='+str(memory),'--property=MemorySwapMax=0','--property=LimitCORE=0','--property=RuntimeMaxSec=1200']
  cmd += ['--setenv='+k+'='+v for k,v in env.items()]
  cmd += [str(binary),'intel',mode]
  with (out/'intel.log').open('x') as log:
   p=subprocess.Popen(cmd,env=child,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   for line in p.stdout:
    if line.startswith('strata_public_preview='):
     text=json.loads(line.split('=',1)[1]);print('Public fixture generation (not saved):\n'+text,flush=True)
    else:
     log.write(line);log.flush()
     if len(line)<600:print(line.rstrip(),flush=True)
   status=p.wait()
  print(json.dumps({'exit':status,'receipt':(out/'intel.json').exists()}),flush=True)
  if status or not (out/'intel.json').exists():raise ValueError('whole port run failed')
 finally:
  helper('host-state.py','resume')
  after=helper('final-secondary-state.py');(out/'host-after.json').write_text(json.dumps(after,indent=2)+'\n')
  good=not after['admission']['state']['admission']['paused'] and after['kernel_fault_counts']==before['kernel_fault_counts'] and not after['nvidia_compute_process_count'] and all(r['used_bytes']<160*1024**2 for r in after['intel']['xe_memory'])
  print(json.dumps({'normal_admission_restored':not after['admission']['state']['admission']['paused'],'faults_unchanged':after['kernel_fault_counts']==before['kernel_fault_counts'],'intel_used_bytes':[r['used_bytes'] for r in after['intel']['xe_memory']],'cleanup_accepted':good}),flush=True)
  if not good:raise ValueError('cleanup evidence failed')
def registration_probe():
 # An explicitly synthetic parser check: no privileged query, model load or GPU.
 build=BASE/sys.argv[2];profile=sys.argv[3]
 binary=next(build.glob('kuno-strata-rust-*'))
 label='language-code-intel-mtp-four-lookahead-down-integer-grouped-ple-gu-stride-shared-head-gpu-host'
 env,_,_=protected.select_trial(label,BASE/'unused-registration-probe.json')
 env.pop('KUNO_LAB_MTP_REPAIR',None)
 if profile=='q2':
  env['KUNO_TEST_MTP']=str(BASE/'kuno-strata-mtp-q2-20261009/mtp-q2_0.gguf')
  env['KUNO_LAB_MTP_PROFILE']='strata-q2';env.pop('KUNO_LAB_MTP_Q8_HEAD',None)
 env.update({'KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES':str(32*1024**3),'KUNO_PUBLIC_MTP_INTEL_USED_BYTES':str(50*1024**2),'KUNO_PUBLIC_MTP_INTEL_DEVICE_ID':str(0xe223)})
 child={k:v for k,v in os.environ.items() if not k.startswith(('KUNO_LAB_','KUNO_TEST_','KUNO_PUBLIC_'))};child.update(env)
 return subprocess.call([str(binary),'--validate-port-registration'],env=child)
if __name__=='__main__':
 if len(sys.argv)==4 and sys.argv[1]=='--registration-only':sys.exit(registration_probe())
 try:run()
 except Exception as e:print(json.dumps({'error_type':type(e).__name__}),flush=True);raise
