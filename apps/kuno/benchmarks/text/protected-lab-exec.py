import os,pathlib,resource,sys
c=next(x.split(":",2)[2].strip() for x in pathlib.Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
p=pathlib.Path("/sys/fs/cgroup")/c.lstrip("/")
if (p/"memory.swap.max").read_text().strip()!="0" or resource.getrlimit(resource.RLIMIT_CORE)!=(0,0):
 raise SystemExit("Protected laboratory cgroup limits missing")
if (p/"memory.max").read_text().strip()!="8589934592":
 raise SystemExit("Protected laboratory memory limit mismatch")
print("verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592",flush=True)
os.execv(sys.argv[1],sys.argv[1:])
