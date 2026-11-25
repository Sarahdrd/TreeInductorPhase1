"""Windows process-tree budgets for future Block 8 adapters; v2, no solver changes."""
import ctypes
from ctypes import wintypes as W
import os
import subprocess
import time

K=ctypes.WinDLL('kernel32',use_last_error=True)
class Basic(ctypes.Structure):
    _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO(ctypes.Structure):
    _fields_=[(n,ctypes.c_uint64) for n in ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
class Extended(ctypes.Structure):
    _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
for name,args,ret in [('CreateJobObjectW',[ctypes.c_void_p,W.LPCWSTR],W.HANDLE),('SetInformationJobObject',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD],W.BOOL),('AssignProcessToJobObject',[W.HANDLE,W.HANDLE],W.BOOL),('TerminateJobObject',[W.HANDLE,W.UINT],W.BOOL),('QueryInformationJobObject',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),('CloseHandle',[W.HANDLE],W.BOOL)]:
    f=getattr(K,name);f.argtypes=args;f.restype=ret

def checked(value):
    if not value:raise ctypes.WinError(ctypes.get_last_error())
    return value

def external(cmd,folder,seconds=120,memory_limit=4*1024**3,env=None):
    started=time.perf_counter();job=checked(K.CreateJobObjectW(None,None));process=None;status='completed';peak=0
    limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000|0x200 # kill-on-close, job-memory hard cap
    limits.JobMemoryLimit=memory_limit
    try:
        checked(K.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)))
        with (folder/'stdout.log').open('w',encoding='utf8') as out,(folder/'stderr.log').open('w',encoding='utf8') as err:
            process=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            checked(K.AssignProcessToJobObject(job,W.HANDLE(int(process._handle))))
            while process.poll() is None:
                if time.perf_counter()-started>=seconds:
                    status='external_timeout';checked(K.TerminateJobObject(job,124));break
                time.sleep(.02)
            process.wait(timeout=5)
            checked(K.QueryInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits),None));peak=limits.PeakJobMemoryUsed
    finally:
        # Closing this owned job terminates every attached descendant even on interruption.
        K.CloseHandle(job)
        if process is not None and process.poll() is None:process.kill();process.wait(timeout=5)
    return dict(external_status=status,returncode=process.returncode,wall_seconds=time.perf_counter()-started,peak_job_memory_bytes=peak,
        process_id=process.pid,timeout_enforcement='Windows job object; kill-on-close; descendant cleanup',memory_note='hard job committed-memory cap; allocation failures appear as nonzero exits, not inferred as solver timeouts')
