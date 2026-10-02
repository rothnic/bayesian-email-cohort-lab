"""Boundary tests for real-v1 replay, independent of prototype count fixtures."""
from dataclasses import replace
import sys
import unittest
import pandas as pd
from replay_v1_ablation import DEFAULT_LAB,freeze,select,forecast_frozen

sys.path.insert(0,str(DEFAULT_LAB))
from cohort_lab.accounting import Snapshot,load_parameters
from cohort_lab.generator import generate_world
from bayes_cohort import forecast_cohort,default_config_dict
from bayes_cohort.model import clear_fit_cache
import json


class V1ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_parameters();p.update(leads_per_cohort=12,cohorts_per_source=3)
        cls.world,_,cls.policy=generate_world(773,'collapse_tail',p)

    def test_thawed_frames_cannot_mutate_archived_bytes(self):
        frozen=freeze(self.world.as_of(10));before=frozen.digest
        view=frozen.thaw(Snapshot)
        self.assertEqual(freeze(view).digest,before)
        view.leads.loc[0,'acquisition_cost']=999.
        self.assertEqual(frozen.digest,before)
        self.assertNotEqual(freeze(view).digest,before)

    def test_future_ledger_append_does_not_change_earlier_prediction(self):
        base=self.world.as_of(10)
        future=self.world.ledger.iloc[:1].copy()
        future['entry_id']='unseen-future-entry';future['observed_at']=100
        future['economic_at']=3;future['amount']=999.
        changed=replace(self.world,ledger=pd.concat([self.world.ledger,future],ignore_index=True)).as_of(10)
        original=freeze(select(base,'middle-w00','same_source',Snapshot))
        later=freeze(select(changed,'middle-w00','same_source',Snapshot))
        self.assertEqual(original.digest,later.digest)
        config=json.dumps(default_config_dict(),sort_keys=True).encode()
        def predict(x):return forecast_frozen(x,self.policy,'middle-w00',config,16,32,Snapshot,forecast_cohort,clear_fit_cache)
        self.assertEqual(predict(original),predict(later))

    def test_target_scope_removes_other_cohort_likelihood_tables(self):
        snap=select(self.world.as_of(17),'middle-w00','target_only',Snapshot)
        for name in ('leads','exposure_daily','engagement_events','ledger','operational_exits'):
            self.assertLessEqual(set(getattr(snap,name).cohort_id),{'middle-w00'})


if __name__=='__main__':unittest.main()
