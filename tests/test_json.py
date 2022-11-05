import unittest

import quanty.json


class TestDeSerialization(unittest.TestCase):
    def test_complex(self):
        cn = complex(1, 1)
        d = quanty.json.dumps(cn)
        cn_ = quanty.json.loads(d)
        self.assertEqual(cn, cn_)

    def test_set(self):
        s = {1, 2, '3', 1+1j}
        d = quanty.json.dumps(s)
        s_ = quanty.json.loads(d)
        self.assertSetEqual(s, s_)


