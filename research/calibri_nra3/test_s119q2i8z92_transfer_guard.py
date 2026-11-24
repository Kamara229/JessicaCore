import unittest
from research.calibri_nra3.s119q2i8z92_transfer_guard import verify_z88,DEV_SHA,DEV_NATIVE_SHA
class TestZ92(unittest.TestCase):
    def test_fail_closed(self):
        for pins in ({},{"run_id":45473255610,"head_sha":"92a1e2bf559e6021989aa592c2eaced5685e79a8","conclusion":"in_progress","artifact_id":1,"joint_sha256":"0"*64},
                     {"run_id":45473255610,"head_sha":"92a1e2bf559e6021989aa592c2eaced5685e79a8","conclusion":"success","artifact_id":1,"joint_sha256":"0"*64}):
            with self.assertRaises(Exception):verify_z88({},pins)
        self.assertEqual(len(DEV_SHA),64)
        self.assertEqual(len(DEV_NATIVE_SHA),64)
if __name__=="__main__":unittest.main()
