import unittest
from dataclasses import replace
import numpy as np
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
        self.assertNotIn('true_contactable',columns)
        self.assertNotIn('true_expected_unique_response_per_lead',columns)
        self.assertNotIn('human_response_multiplier',columns)
        self.assertEqual(len(next(iter(self.truth.cohort_paths.values()))),426)

    def test_cohort_retention_varies_and_is_monotone(self):
        registry=self.world.cohort_registry
        cohort_ids=registry.loc[registry.source_id=='established_a','cohort_id'].tolist()
        paths=self.truth.cohort_truth
        pair=paths[paths.cohort_id.isin(cohort_ids[:2])]
        first=pair[pair.cohort_id==cohort_ids[0]].sort_values('age')
        second=pair[pair.cohort_id==cohort_ids[1]].sort_values('age')
        self.assertEqual(first.true_contactable_fraction.iloc[0],1.)
        self.assertTrue(first.true_contactable_fraction.is_monotonic_decreasing)
        self.assertTrue((first.true_contactable_fraction.to_numpy() !=
                         second.true_contactable_fraction.to_numpy()).any())
        self.assertNotEqual(self.truth.parameters.true_contactable_half_life.iloc[0],
                            self.truth.parameters.true_contactable_half_life.iloc[1])

    def test_cpc_squeeze_preserves_people_and_clicks(self):
        fixed,fixed_truth=generate(710,'fixed_price',leads=100,last_birth=14)
        squeezed,squeezed_truth=generate(710,'cpc_squeeze',leads=100,last_birth=14)
        cohort='established_a-b000'
        a=fixed.observations.query('cohort_id == @cohort').sort_values('age')
        b=squeezed.observations.query('cohort_id == @cohort').sort_values('age')
        for field in ('accepted_exposure','raw_events','raw_unique_responders','net_human_count'):
            self.assertTrue((a[field].to_numpy()==b[field].to_numpy()).all(),field)
        oracle=squeezed_truth.cohort_truth.query('cohort_id == @cohort').sort_values('age')
        reference=fixed_truth.cohort_truth.query('cohort_id == @cohort').sort_values('age')
        np.testing.assert_array_equal(oracle.true_net_cpc.to_numpy()[:42],
                                      reference.true_net_cpc.to_numpy()[:42])
        self.assertLess(oracle.true_net_cpc.iloc[-1],reference.true_net_cpc.iloc[-1])

    def test_complete_week_and_month_rollups_are_person_deduplicated(self):
        rollups=self.truth.cohort_rollups
        self.assertEqual(set(rollups.window_days.unique()),{7,30})
        self.assertTrue((rollups.window_end_age-rollups.window_start_age+1==rollups.window_days).all())
        self.assertTrue((rollups.distinct_human_responders<=rollups.eligible_at_start).all())
        self.assertTrue((rollups.contactable_at_end<=rollups.acquired_leads).all())
        self.assertTrue((self.world.observations.raw_events>=self.world.observations.raw_unique_responders).all())


if __name__=='__main__':unittest.main()
