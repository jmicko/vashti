#!/usr/bin/env python3
"""Refresh already-authorized sudo, never request/read a password. Bounded lifetime.
Run from the shared terminal after authentication; stop when the work ends.
"""
import argparse,os,pathlib,signal,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--minutes',type=int,default=240);p.add_argument('--pid-file',type=pathlib.Path,required=True);a=p.parse_args()
if not 1<=a.minutes<=240:raise SystemExit('bounded 1..240 minutes')
def refresh():
 return subprocess.call(['sudo','-n','-v'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)==0
if not refresh():raise SystemExit('sudo cache unavailable; no password prompt was opened')
a.pid_file.write_text(str(os.getpid())+'\n')
signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(0)))
try:
 deadline=time.monotonic()+a.minutes*60
 while time.monotonic()<deadline:
  time.sleep(90)
  if not refresh():break
finally:
 if a.pid_file.exists() and a.pid_file.read_text().strip()==str(os.getpid()):a.pid_file.unlink()
