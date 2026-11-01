"""Independent exhaustive oracles and byte/quality/corruption regression tests."""
import itertools
import random
import unittest
import numpy as np
from shared_cost import select_formats
from precision_frontier import Candidate, allocate
from residual_page import pack_page, unpack_page
from hybrid_codec import encode_hybrid_bundle, decode_hybrid
from dual_codec import StaticPairDictionary


class SharedCostTests(unittest.TestCase):
    def test_local_win_global_loss(self):
        audit = select_formats([1000], [900], 524288)
        self.assertEqual(audit['modes'], ('CL8',))
        self.assertEqual(audit['total_bytes'], 1000)

    def test_shared_cost_paid_once_and_tie_prefers_base(self):
        self.assertEqual(select_formats([1000]*3, [600]*3, 1000)['total_bytes'], 2800)
        self.assertEqual(select_formats([1000]*2, [500]*2, 1000)['modes'], ('CL8','CL8'))

    def test_exhaustive_oracle(self):
        rng = random.Random(42)
        for _ in range(100):
            a = [rng.randrange(1,1000) for _ in range(6)]
            b = [rng.randrange(1,1000) for _ in range(6)]
            opening = rng.randrange(1500)
            optimum = min(sum(b[i] if bit else a[i] for i,bit in enumerate(bits))
                          + opening*any(bits) for bits in itertools.product((0,1), repeat=6))
            self.assertEqual(select_formats(a,b,opening)['total_bytes'], optimum)

    def test_invalid_costs(self):
        for bad in [-1, 1.5, True, float('nan')]:
            with self.assertRaises(ValueError):
                select_formats([bad],[1],0)

    def test_physical_streams(self):
        d = StaticPairDictionary()
        matrices = [np.zeros((4,64),np.int8), np.ones((3,32),np.int8)]
        blobs,audit = encode_hybrid_bundle(matrices,d,external_bytes=28)
        self.assertEqual(audit['total_bytes'],sum(map(len,blobs))+audit['dictionary_bytes']+28)
        self.assertLessEqual(audit['total_bytes'],audit['all_base_bytes'])
        for q,blob in zip(matrices,blobs):
            np.testing.assert_array_equal(q,decode_hybrid(blob,d))
        with self.assertRaises(ValueError):
            encode_hybrid_bundle([np.full((2,4),256)],d)


class FrontierTests(unittest.TestCase):
    def test_exhaustive_oracle_with_shared_dictionary_and_two_strata(self):
        rng = random.Random(84)
        for _ in range(40):
            pages = [[Candidate(str(j),rng.randrange(10,100),
                                (rng.randrange(10),rng.randrange(10)),bool(j%2))
                      for j in range(3)] for _ in range(4)]
            limits = (20,22)
            feasible = []
            for choices in itertools.product(*pages):
                if all(sum(c.errors[i] for c in choices)<=limits[i] for i in range(2)):
                    feasible.append(sum(c.byte_count for c in choices)
                                    + 61*any(c.needs_dictionary for c in choices))
            if feasible:
                self.assertEqual(allocate(pages,limits,61).byte_count,min(feasible))
            else:
                with self.assertRaises(ValueError): allocate(pages,limits,61)

    def test_rare_stratum_cannot_be_hidden_by_mean(self):
        candidates = [[Candidate('cheap',10,(0.,10.)),Candidate('safe',20,(1.,1.))]]
        self.assertEqual(allocate(candidates,(2.,2.)).choices,('safe',))

    def test_fail_closed_and_state_limit(self):
        with self.assertRaises(ValueError):
            allocate([[Candidate('bad',1,(3.,))]],(2.,))
        with self.assertRaises(RuntimeError):
            allocate([[Candidate('a',1,(2.,)),Candidate('b',2,(1.,))]],(3.,),max_states=1)
        with self.assertRaises(ValueError):
            allocate([[Candidate('bad',1,(float('nan'),))]],(2.,))


class ResidualTests(unittest.TestCase):
    def test_full_residual_accuracy_and_corruption(self):
        rng = np.random.default_rng(53)
        w = rng.normal(size=(8,64))
        blob = pack_page(w,np.ones(64),.885,1.)
        reconstructed = unpack_page(blob)
        self.assertLess(np.max(np.abs(w-reconstructed)),.002)
        with self.assertRaises(ValueError): unpack_page(blob[:-1])
        damaged=bytearray(blob);damaged[-1]^=1
        with self.assertRaises(ValueError): unpack_page(damaged)

    def test_disjoint_row_error_adds(self):
        rng = np.random.default_rng(54)
        w = rng.normal(size=(12,32)); x=rng.normal(size=(24,32))
        decoded=[unpack_page(pack_page(page,np.ones(32),.75,.05))
                 for page in np.split(w,3)]
        local=sum(np.sum((x@(a-b).T)**2) for a,b in zip(np.split(w,3),decoded))
        full=np.sum((x@(w-np.vstack(decoded)).T)**2)
        self.assertAlmostEqual(local,full,places=8)

    def test_empty_and_invalid_inputs(self):
        with self.assertRaises(ValueError): pack_page(np.empty((0,4)),np.ones(4),.8,0.)
        with self.assertRaises(ValueError): pack_page(np.ones((2,4)),np.ones(4),.8,1.1)


if __name__ == '__main__': unittest.main()
