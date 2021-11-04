from scipy.optimize import fsolve
import numpy as np


class Hamiltonian:
    def __init__(self, Npw, L, Znuc, beta2=1.5**2, pos=0.625):
        self.Npw = Npw
        self.L = L
        self._Gvec = None
        self._Gvec_2 = None
        self._coulomb = None
        self.pos = pos * L
        self.beta2 = beta2
        self.Znuc = Znuc
        self.Nel = Znuc
        self._rhonuc_G = None
        self._rho = None
        self._rho_G = None

    @property
    def rhonuc_G(self):
        if self._rhonuc_G is None:
            self._rhonuc_G = -(self.Znuc / self.L) * np.exp(
                1j * self.Gvec_2 * self.pos
            ) * np.exp(-0.5 * self.Gvec_2**2 * self.beta2)
        return self._rhonuc_G

    @property
    def coulomb(self):
        if self._coulomb is None:
            self._coulomb = np.zeros(len(self.Gvec_2))
            cond = np.abs(self.Gvec_2) > 1e-10
            self._coulomb[cond] = 4 * np.pi / self.Gvec_2[cond]**2
        return self._coulomb

    @property
    def Gvec(self):
        if self._Gvec is None:
            self._Gvec = np.fft.fftfreq(self.Npw, d=self.dL) * (2 * np.pi)
        return self._Gvec

    @property
    def Gvec_2(self):
        if self._Gvec_2 is None:
            self._Gvec_2 = np.fft.fftfreq(2 * self.Npw, d=self.dL_2) * (2 * np.pi)
        return self._Gvec_2

    @property
    def dL(self):
        return self.L / self.Npw

    @property
    def dL_2(self):
        return self.dL / 2

    @property
    def _G_indices(self):
        ni = np.roll(np.arange(self.Npw)+1-self.Npw//2, self.Npw//2+1)
        return ni[:,None]-ni

    @property
    def hamMat(self):
        return np.fft.ifft(self.veff)[self._G_indices] + 0.5 * self.Gvec**2 * np.eye(self.Npw)

    def computeRho(self, vals, vecs, Nel, kT):
        # find the Fermi energy
        beta = 1. / kT
        fermi = lambda x: 1/(1 + np.exp(beta * x)) if (beta * x) < 50 else 0
        target = lambda mu: sum(fermi(eps-mu) for eps in vals) - Nel
        mu, = fsolve (target, x0 = vals[Nel])
        self.mu = mu
        # compute rho and kinetic energy
        self.Ekin = 0.
        rho=np.zeros(shape=2*self.Npw, dtype=np.float64)
        Npw2 = int (self.Npw/2)
        for i in np.ndindex(vals.shape):
            focc = fermi(vals[i]-mu)
            if focc > 1e-12:
                # get psi on large FFT mesh
                psi_expand = np.zeros(shape=(2*self.Npw),dtype=np.complex128)
                psi_expand[0:Npw2] = vecs[0:Npw2,i].flatten ()
                psi_expand[-Npw2:] = vecs[-Npw2:,i].flatten ()
                psi = np.fft.fft(psi_expand)/np.sqrt(self.L)
                # add to density
                rho += focc * (psi.real **2 + psi.imag ** 2)
                # compute kinetic energy contribution
                self.Ekin += focc * 0.5 * sum (
                            (c.real ** 2 + c.imag ** 2) * g ** 2
                             for c,g in zip(vecs[:,i].flatten (),
                                            self.Gvec)
                            )
        return rho

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho_in):
        self._rho = np.asarray(rho_in)
        self._rho_G = None

    @property
    def rho_G(self):
        if self._rho_G is None:
            self._rho_G = np.fft.ifft(self.rho)
            self._rho_G += self.rhonuc_G
            self._rho_G[0] = 0.
        return self._rho_G

    @property
    def alpha(self):
        return -3/4 * (3/np.pi)**(1/3)

    @property
    def eXc(self):
        return np.sum(self.rho[self.rho > 0]**(4./3.)) * self.alpha * self.dL_2

    @property
    def vXc(self):
        return (4./3.) * self.alpha * np.maximum(self.rho, 0)**(1./3.)

    @property
    def V_G(self):
        return self.rho_G * self.coulomb

    @property
    def vH(self):
        return np.real (np.fft.fft(self.V_G))

    @property
    def veff(self):
        veff = self.vH + self.vXc
        if 'v_loc' in self.__dir__():
            veff += self.v_loc
        return veff

    @property
    def eLoc(self):
        if 'v_loc' in self.__dir__():
            return sum (self.v_loc * self.rho) * self.dL_2
        else:
            return 0.

    @property
    def E_H(self):
        return 0.5 * np.vdot(self.rho_G, self.V_G).real * self.L

    def computePot(self, rho):
        self.rho = rho

    def getEnergy (self):
        return self.Ekin + self.eXc + self.E_H + self.eLoc
