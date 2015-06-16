from control4.core.cpd import CPD,FactoredCategoricalDistribution,DiagonalGaussian
import theano.tensor as TT


class QuakeActionDistribution(CPD):
    """
    LOOK: R2
    WEAPON: Z10
    FWD/BACK: Z3
    LEFT/RIGHT: Z3
    CROUCH/JUMP: Z3
    ACT/NOACT: Z2
    """

    def __init__(self):
        self._look = DiagonalGaussian(2)
        self._discrete = FactoredCategoricalDistribution([10,3,3,3,2])

    def mls(self, a_na):
        return TT.concatenate([self._look.mls(a_na[:,:2]), self._discrete.mls(a_na[:,2:])],axis=1)

    def liks(self, a_na, b_nb):
        return self._look.liks(a_na[:,:2]) * self._discrete.liks(a_na[:,2:])

    def logliks(self, a_na, b_nb):
        return self._look.logliks(a_na[:,:2]) + self._discrete.logliks(a_na[:,2:])

    def draw(self, a_na):
        return TT.concatenate([self._look.draw(a_na[:,:2]), self._discrete.draw(a_na[:,2:])],axis=1)

    def draw_numeric(self, a_na):
        return TT.concatenate([self._look.draw_numeric(a_na[:,:2]), self._discrete.draw_numeric(a_na[:,2:])],axis=1)

    def draw_frozen(self, a_na):
        return TT.concatenate([self._look.draw_frozen(a_na[:,:2]), self._discrete.draw_frozen(a_na[:,2:])],axis=1)

    def draw_frozen_numeric(self, a_na):
        return TT.concatenate([self._look.draw_frozen_numeric(a_na[:,:2]), self._discrete.draw_frozen_numeric(a_na[:,2:])],axis=1)

    def entropy(self, a_na):
        return self._look.entropy(a_na[:,:2]) + self._discrete.entropy(a_na[:,2:])

    def kl(self, a_na):
        return self._look.kl(a_na[:,:2]) + self._discrete.kl(a_na[:,2:])

    def a_dim(self):
        return self._look.a_dim() + self._discrete.a_dim()

    def b_dim(self):
        return self._look.b_dim() + self._discrete.b_dim()

    def r_dim(self):
        return self._look.r_dim() + self._discrete.r_dim()

    def fvp(self, a_na, v_na):
        return self._look.fvp(a_na, v_na) + self._discrete.fvp(a_na, v_na)
