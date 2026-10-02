import unittest
import numpy as np
from cohort_lab.generator import generate_world
from cohort_lab.accounting import Policy, ledger_path, summarize_path

class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world,cls.truth,cls.policy=generate_world(11,"collapse_tail",small=True)
    def test_generator_is_reproducible(self):
        world,truth,policy=generate_world(11,"collapse_tail",small=True)
        self.assertTrue(world.ledger.equals(self.world.ledger))
        self.assertTrue(world.engagement_events.equals(self.world.engagement_events))
        self.assertEqual(truth.parameters["leads_per_cohort"],80)
        self.assertEqual(truth.parameters["cohorts_per_source"],6)
    def test_ledger_reconciles_and_is_immutable(self):
        world=self.world
        self.assertTrue(world.ledger.entry_id.is_unique)
        self.assertAlmostEqual(world.ledger.amount.sum(),self.truth.complete_ledger.amount.sum())
        for cohort in world.leads.cohort_id.unique():
            path=self.truth.cohort_path(world.leads,cohort,self.policy)
            amount=self.truth.complete_ledger.loc[self.truth.complete_ledger.cohort_id.eq(cohort),"amount"].sum()
            self.assertAlmostEqual(path[-1],amount,places=9)
        revisions=world.ledger.loc[world.ledger.revises_entry_id.ne("")]
        self.assertTrue(revisions.revises_entry_id.isin(world.ledger.entry_id).all())
    def test_asof_no_future_leakage(self):
        original=self.world.ledger.copy(deep=True)
        for cutoff in (3,7,14,30,60):
            snap=self.world.as_of(cutoff)
            self.assertTrue(snap.leads.acquired_at.le(cutoff).all())
            for frame,time in ((snap.ledger,"economic_at"),(snap.engagement_events,"occurred_at"),(snap.exposure_daily,"date"),(snap.operational_exits,"occurred_at")):
                self.assertTrue(frame[time].le(cutoff).all())
                self.assertTrue(frame.observed_at.le(cutoff).all())
            self.assertTrue(snap.engagement_events.event_id.is_unique)
        self.assertTrue(self.world.ledger.equals(original))
    def test_early_invoice_revision_excluded(self):
        self.assertFalse(self.world.as_of(3).ledger.kind.eq("acquisition_revision").any())
        self.assertTrue(self.world.as_of(30).ledger.kind.eq("acquisition_revision").any())
    def test_classification_revision_visibility(self):
        revised=self.world.engagement_events.loc[self.world.engagement_events.revision_sequence.eq(1)].iloc[0]
        initial=self.world.engagement_events.loc[(self.world.engagement_events.event_id.eq(revised.event_id)) & self.world.engagement_events.revision_sequence.eq(0)].iloc[0]
        early=self.world.as_of(int(initial.observed_at)).engagement_events
        late=self.world.as_of(int(revised.observed_at)).engagement_events
        self.assertEqual(early.loc[early.event_id.eq(revised.event_id),"qualification"].iloc[0],"human")
        self.assertEqual(late.loc[late.event_id.eq(revised.event_id),"qualification"].iloc[0],"bot")
    def test_inactivity_is_not_operational_death(self):
        snap=self.world.as_of(30)
        clickers=set(snap.engagement_events.loc[snap.engagement_events.qualification.eq("human"),"lead_id"])
        exited=set(snap.operational_exits.lead_id)
        quiet=set(snap.leads.lead_id)-clickers-exited
        self.assertGreater(len(quiet),0)
        latest=snap.exposure_daily.groupby(["cohort_id","arm"]).tail(1)
        self.assertGreater(latest.eligible.sum(),len(clickers))
    def test_never_vs_later_and_reversals(self):
        path=np.full(426,-1.);path[200:]=1.
        later=summarize_path(path)[-1]
        self.assertEqual(later["payback_state"],"later_than_180")
        never=summarize_path(np.full(426,-1.))[-1]
        self.assertEqual(never["payback_state"],"never_under_policy")
        self.assertIsNone(never["first_payback_age"])
        path[210]=-2.
        self.assertFalse(summarize_path(path)[-1]["remains_positive_after_payback"])
    def test_policy_bounds(self):
        w=self.world
        dates=w.leads.groupby("cohort_id").acquired_at.first()
        self.assertTrue((w.exposure_daily.date-w.exposure_daily.cohort_id.map(dates)).le(self.policy.last_send_age).all())
        self.assertTrue((w.ledger.economic_at-w.ledger.cohort_id.map(dates)).le(self.policy.terminal_age).all())
        self.assertTrue((w.ledger.observed_at-w.ledger.cohort_id.map(dates)).le(self.policy.terminal_age).all())
        human=self.truth.final_human_events
        self.assertTrue((human.occurred_at-human.send_at).le(30).all())
        self.assertTrue((human.occurred_at>human.send_at).any())
    def test_shared_calendar_and_sparse_size(self):
        self.assertTrue(self.truth.calendar_shocks.date.is_unique)
        self.assertEqual(len(self.world.exposure_daily),18*2*366)
        self.assertLess(len(self.world.engagement_events),18*80*366//10)
        self.assertTrue(self.world.engagement_events.message_id.duplicated().any())

if __name__=="__main__":unittest.main()
