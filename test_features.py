import unittest
import numpy as np
from features import encode

class FeatureTests(unittest.TestCase):
    def hands(self):
        h=np.zeros((21,3),np.float32)
        for i in range(21):h[i]=[.2+i*.003,.3+i*.008,i*.001]
        return [h,h+np.array([.3,.05,0])]
    def test_order_independence(self):
        hands=self.hands()
        np.testing.assert_allclose(encode(hands),encode(hands[::-1]),atol=1e-5)
    def test_translation_invariance(self):
        hands=self.hands();moved=[h+np.array([.2,.1,0]) for h in hands]
        np.testing.assert_allclose(encode(hands),encode(moved),atol=1e-5)
    def test_scale_invariance(self):
        hands=self.hands()
        np.testing.assert_allclose(encode(hands),encode([h*2 for h in hands]),atol=1e-5)
    def test_shape_and_hand_count(self):
        x=encode(self.hands());self.assertEqual(x.shape,(128,));self.assertEqual(list(x[-2:]),[0.,1.])
        x=encode(self.hands()[:1]);self.assertEqual(list(x[-2:]),[1.,0.]);self.assertTrue(np.isfinite(x).all())
if __name__=='__main__':unittest.main()
