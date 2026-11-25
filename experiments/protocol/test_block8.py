"""Bounded adapter infrastructure checks; no canonical models fitted."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from .block8 import Representation,predict_tree,POLICY
from .block8_process import external
from .checkpoints import Store,writer_lock,IntegrityError

class Block8Tests(unittest.TestCase):
    def test_train_only_missing_constant_and_unseen(self):
        train=pd.DataFrame({'n':[1.,2.,np.nan,4.],'cat':['a','b',None,'a'],'constant':[1,1,1,1],'empty':[np.nan]*4})
        valid=pd.DataFrame({'n':[1000.],'cat':['unknown'],'constant':[2],'empty':[99.]})
        for alg in ['murtree','quant_bnb']:
            p=Representation(alg).fit(train);before=json.dumps(p.spec);a=p.transform(train);b=p.transform(valid)
            self.assertEqual(before,json.dumps(p.spec));self.assertEqual(p.spec[0]['fill'],2.);self.assertEqual(a.shape[1],b.shape[1]);self.assertTrue(np.isfinite(a).all());self.assertTrue((b[0,-2:]==0).all())
            if alg=='murtree':self.assertTrue(np.isin(a,[0,1]).all())
            else:self.assertEqual(b[0,0],1000.)
    def test_threshold_cap_and_original_target_restoration(self):
        x=pd.DataFrame({'x':np.arange(100.)});p=Representation('murtree').fit(x);self.assertEqual(len(p.spec[0]['cuts']),16)
        tree={'feature':0,'left':{'label':0},'right':{'feature':1,'left':{'label':1},'right':{'label':2}}}
        codes=predict_tree(tree,np.array([[0,0],[1,0],[1,1]]));np.testing.assert_array_equal(np.array(['classA','classB','classC'])[codes],['classA','classB','classC'])
    def test_external_timeout_and_errors(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);r=external([sys.executable,'-c','import time;time.sleep(10)'],folder,seconds=.2)
            self.assertEqual(r['external_status'],'external_timeout');self.assertLess(r['wall_seconds'],6)
            r=external([sys.executable,'-c','raise RuntimeError("test-only")'],folder,seconds=5);self.assertNotEqual(r['returncode'],0)
    def test_descendant_cleanup(self):
        import ctypes
        from ctypes import wintypes as W
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);pidfile=folder/'child.pid'
            code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],creationflags=subprocess.CREATE_NO_WINDOW);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)"
            r=external([sys.executable,'-c',code,str(pidfile)],folder,seconds=.8)
            self.assertEqual(r['external_status'],'external_timeout');self.assertTrue(pidfile.exists())
            k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenProcess.restype=W.HANDLE
            k.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];k.CloseHandle.argtypes=[W.HANDLE]
            handle=k.OpenProcess(0x100000,False,int(pidfile.read_text()))
            if handle:
                try:self.assertEqual(k.WaitForSingleObject(handle,1000),0)
                finally:k.CloseHandle(handle)
    def test_julia_timeout(self):
        from .block8 import BASE
        with tempfile.TemporaryDirectory() as t:
            r=external([str(BASE/'sidecars/julia-1.10.10/bin/julia.exe'),'--startup-file=no','-e','sleep(30)'],Path(t),seconds=.2)
            self.assertEqual(r['external_status'],'external_timeout');self.assertLess(r['wall_seconds'],3)
    def test_checkpoint_failure_then_success(self):
        with tempfile.TemporaryDirectory() as t:
            s=Store(t,{'version':'test'});k=('toy','murtree',42,1,1)
            with writer_lock(s.root):
                s.initialize();s.save(k,dict(attempt=1,execution_status='failure',status='external_timeout'))
                s.save(k,dict(attempt=2,execution_status='success',status='feasible_uncertified',optimal=False))
                with self.assertRaises(IntegrityError):s.save(k,dict(attempt=3,execution_status='success'))
            self.assertEqual(len(s.attempts(k)),2)
    def test_no_full_run_enabled(self):
        self.assertFalse(POLICY['full_execution_enabled']);self.assertEqual(POLICY['depth'],2)

if __name__=='__main__':unittest.main()
