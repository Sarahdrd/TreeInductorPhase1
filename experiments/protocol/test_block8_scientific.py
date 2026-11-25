"""Lightweight analysis tests; no solver, fitting or split generation."""
import unittest
import pandas as pd
from .block8_scientific import paired
from .checkpoints import IntegrityError


class PairedTests(unittest.TestCase):
    def records(self):
        common=dict(dataset='example',seed=42,repeat=1,split_id='split',
                    dataset_fingerprint='data',environment_id='env',representation_id='rep',configuration_id='config')
        return pd.DataFrame([dict(common,model=m,fold=f,validation_macro_f1=score)
                             for m,vals in [('murtree',[.5,.7]),('cart',[.4,.8])]
                             for f,score in enumerate(vals,1)])

    def test_pairs_align_by_keys_not_row_order(self):
        frame=self.records()
        result=paired(frame,frame[frame.model.eq('cart')].iloc[::-1],
                      'murtree','matched',True)
        self.assertAlmostEqual(result.delta.mean(),0.)
        self.assertAlmostEqual(result.delta.iloc[0],.1)

    def test_representation_mismatch_rejected(self):
        frame=self.records();cart=frame[frame.model.eq('cart')].copy()
        cart['representation_id']='other'
        with self.assertRaises(IntegrityError):paired(frame,cart,'murtree','matched',True)

    def test_missing_fold_rejected(self):
        frame=self.records()
        with self.assertRaises(IntegrityError):
            paired(frame,frame[frame.model.eq('cart')].iloc[:1],'murtree','matched',True)

    def test_duplicate_fold_rejected(self):
        frame=self.records();cart=frame[frame.model.eq('cart')]
        with self.assertRaises(IntegrityError):
            paired(frame,pd.concat([cart,cart.iloc[:1]]),'murtree','matched',True)


if __name__=='__main__':unittest.main()
