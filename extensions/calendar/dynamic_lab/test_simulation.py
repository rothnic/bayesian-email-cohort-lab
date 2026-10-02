import unittest
from dataclasses import replace
import pandas as pd
from .simulation import generate


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.world,cls.truth=generate(31,'cheap_bad',leads=10,last_birth=84)

    def test_delayed_audits_and_settlement_are_missing(self):
        f=self.world.as_of(3).frame()
        self.assertTrue(f.audit_n.isna().all())
        self.assertTrue(f.net_human_revenue.isna().all())
        self.assertTrue((f.payout_complete_fraction==0).all())

    def test_immutable_asof_and_future_append(self):
        snap=self.world.as_of(30);f=snap.frame();f.loc[0,'raw_events']=999
        self.assertNotEqual(f.loc[0,'raw_events'],snap.frame().loc[0,'raw_events'])
        late=self.world.observations.iloc[:1].copy();late.observed_at=100;late.raw_events=999
        altered=replace(self.world,observations=pd.concat([self.world.observations,late],ignore_index=True))
        self.assertEqual(altered.as_of(30).digest,snap.digest)

    def test_new_source_not_visible_before_launch(self):
        self.assertFalse((self.world.as_of(69).frame().source_id=='new_cheap').any())
        self.assertGreater(self.world.quote_at('new_cheap',69),0)
        self.assertTrue((self.world.as_of(79).frame().source_id=='new_cheap').any())

    def test_future_cpl_does_not_reprice_old_cohort(self):
        world,_=generate(31,'rising_cpl',leads=10,last_birth=84)
        c0=next(t for t in world.targets(0) if t['source_id']=='established_a')
        c1=next(t for t in world.targets(84) if t['cohort_id']==c0['cohort_id'])
        self.assertEqual(c0['acquisition_cost'],c1['acquisition_cost'])
        self.assertGreater(world.quote_at('established_a',84),world.quote_at('established_a',0))

    def test_truth_not_in_observation_snapshot(self):
        columns=set(self.world.as_of(84).frame().columns)
        self.assertNotIn('true_human_events',columns)
        self.assertNotIn('true_bot_events',columns)
        self.assertNotIn('human_response_multiplier',columns)
        self.assertEqual(len(next(iter(self.truth.cohort_paths.values()))),426)


if __name__=='__main__':unittest.main()
