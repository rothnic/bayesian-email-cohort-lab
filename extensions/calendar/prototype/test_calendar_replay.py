import dataclasses
import math
import unittest
from calendar_replay import (record,fixture,as_of,scoped,calendar_paths,joint_scenarios,
                             contribution_path,first_payback,replay_manifest,DynamicHierarchy)


class ReplayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.stream=fixture(scenario='cheap_bad')

    def test_later_observations_leave_earlier_input_bytes_unchanged(self):
        original=as_of(self.stream,30)
        extra=record('late-correction','response_cell','established_a','established_a:b0',2,75,
                     replaces='cell:established_a:b0:2',revision=7,raw_events=900)
        self.assertEqual(original.canonical_bytes(),as_of(self.stream+(extra,),30).canonical_bytes())
        self.assertNotEqual(as_of(self.stream,75).digest,as_of(self.stream+(extra,),75).digest)

    def test_future_effective_fact_excluded_even_if_reported_early(self):
        future=record('future','contract','','',80,20,human_net_value=.8)
        self.assertEqual(as_of((),30),as_of((future,),30))

    def test_missing_report_is_not_a_zero_count(self):
        missing=record('missing','response_cell','s','c',0,None,raw_events=11)
        self.assertEqual(as_of((missing,),100).records,())

    def test_revision_replaces_measurement_not_signed_payment(self):
        orig=record('x','response_cell','s','c',1,2,raw_events=9)
        rev=record('x1','response_cell','s','c',1,5,replaces='x',revision=1,raw_events=4)
        receipt=record('pay','payment','s','c',1,3,amount=1.)
        reversal=record('rev','payment','s','c',1,6,amount=-.3)
        self.assertEqual(as_of((orig,rev),3).records,(orig,))
        self.assertEqual(as_of((orig,rev),5).records,(rev,))
        self.assertAlmostEqual(sum(r.get('amount') for r in as_of((receipt,reversal),6).records),.7)
        with self.assertRaises(ValueError):
            record('bad','payment','s','c',1,7,replaces='pay',amount=-.3)

    def test_snapshot_and_payload_are_immutable_and_hash_order_invariant(self):
        snap=as_of(self.stream,30)
        self.assertEqual(snap.digest,as_of(tuple(reversed(self.stream)),30).digest)
        with self.assertRaises(dataclasses.FrozenInstanceError):snap.cutoff=4
        with self.assertRaises(TypeError):snap.records[0].values[0]=('x',1)

    def test_chained_revision_resolves_to_one_measurement(self):
        x=record('x','response_cell','s','c',1,2,raw_events=9)
        x1=record('x1','response_cell','s','c',1,5,replaces='x',revision=1,raw_events=4)
        x2=record('x2','response_cell','s','c',1,8,replaces='x1',revision=2,raw_events=3)
        self.assertEqual(as_of((x,x1,x2),8).records,(x2,))
        self.assertEqual(as_of((x,x1,x2),5).records,(x1,))

    def test_new_source_enters_only_at_announced_acquisition(self):
        self.assertFalse(any(r.source=='new_cheap' for r in as_of(self.stream,41).records))
        self.assertTrue(any(r.source=='new_cheap' for r in as_of(self.stream,42).kinds('cohort')))

    def test_old_purchase_cost_does_not_follow_new_quotes(self):
        stream=fixture(scenario='squeeze')
        def cost(t):return next(r.get('acquisition_cost') for r in as_of(stream,t).kinds('cohort') if r.cohort=='established_a:b0')
        self.assertEqual(cost(0),cost(112))
        quotes=[r for r in as_of(stream,84).kinds('quote') if r.source=='established_a']
        self.assertGreater(max(r.get('price_per_lead') for r in quotes),.20)

    def test_information_scopes_separate_targets_sources_and_global(self):
        snap=as_of(self.stream,60)
        counts=[len(scoped(snap,'established_a:b0',scope).kinds('cohort')) for scope in ('target_only','same_source','all_sources')]
        self.assertEqual(counts,[1,9,21])

    def test_audits_remain_missing_until_report_and_events_can_repeat(self):
        self.assertEqual(as_of(self.stream,3).kinds('audit'),())
        self.assertTrue(as_of(self.stream,14).kinds('audit'))
        self.assertTrue(any(r.get('classifier_human_events')>r.get('distinct_reported_humans') for r in as_of(self.stream,40).kinds('response_cell')))

    def test_all_cohorts_share_each_future_calendar_draw(self):
        paths=calendar_paths(seed=2,draws=8,horizon=30)
        joint=joint_scenarios(('a','b','c'),paths)
        for i,draw in enumerate(joint):
            for _,path in draw:self.assertIs(path,paths[i])
        self.assertNotEqual(paths[0],paths[1])

    def test_accounting_monotonicity_and_no_payback_draws(self):
        receipts=(0.,.1,.3,.5); costs=(.02,)*4
        base=contribution_path(.5,receipts,costs)
        cheaper=contribution_path(.2,receipts,costs)
        higher=contribution_path(.5,tuple(2*x for x in receipts),costs)
        self.assertLessEqual(first_payback(cheaper),first_payback(base))
        self.assertLessEqual(first_payback(higher),first_payback(base))
        paths=(base,contribution_path(10,receipts,costs))
        cdf=sum(first_payback(path)<=3 for path in paths)/len(paths)
        self.assertEqual(cdf,.5)
        self.assertEqual(first_payback(paths[1]),math.inf)

    def test_replay_uses_same_fixed_prior_and_cannot_fake_inference(self):
        rows=replay_manifest(self.stream,[3,14,60],'established_a:b0')
        self.assertEqual(len({r['prior_sha256'] for r in rows}),1)
        self.assertEqual({r['forecast_status'] for r in rows},{'not_implemented'})
        with self.assertRaises(NotImplementedError):DynamicHierarchy().predict(as_of(self.stream,3),'established_a:b0',b'fixed',1)


if __name__=='__main__':unittest.main()
