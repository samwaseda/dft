from scipy.optimize import fsolve
import numpy as np


class Hamiltonian:
    """
    Simple plane wave DFT class.

    Example:
    ```
    h = Hamiltonian(Npw=100, L=5, Nel=7)
    for _ in range(10):
        h.rho = rho_in
        rho_out = h.get_rho()
        rho_in = rho_out
    ```
    """
    def __init__(self, Npw, L, Znuc, beta2=1.5**2, pos=0.625, kT=1e-3, min_focc=1e-12):
        """
        Args:
            Npw (int): Number of plane waves
            L (float): Box length
            Znuc (int): Number of protons
            beta2 (float): Gaussian width of the core
            pos (float): Relative position of the core
            kT (float): Boltzmann factor times temperature
            min_focc (float): Minimum occupancy to consider for the energy calculation
        """
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
        self.kT = kT
        self._mu = None
        self._vals = None
        self._vecs = None
        self._min_focc = min_focc

    @property
    def rhonuc_G(self):
        if self._rhonuc_G is None:
            self._rhonuc_G = -(self.Znuc / self.L) * np.exp(
                1j * self._Gvec_2 * self.pos
            ) * np.exp(-0.5 * self._Gvec_2**2 * self.beta2)
        return self._rhonuc_G

    @property
    def coulomb(self):
        if self._coulomb is None:
            self._coulomb = np.zeros(len(self._Gvec_2))
            cond = np.abs(self._Gvec_2) > 1e-10
            self._coulomb[cond] = 4 * np.pi / self._Gvec_2[cond]**2
        return self._coulomb

    @property
    def _Gvec(self):
        if self._Gvec is None:
            self._Gvec = np.fft.fftfreq(self.Npw, d=self._dL) * (2 * np.pi)
        return self._Gvec

    @property
    def _Gvec_2(self):
        if self._Gvec_2 is None:
            self._Gvec_2 = np.fft.fftfreq(2 * self.Npw, d=self._dL / 2) * (2 * np.pi)
        return self._Gvec_2

    @property
    def _dL(self):
        return self.L / self.Npw

    @property
    def _G_indices(self):
        ni = np.roll(np.arange(self.Npw) + 1 - self.Npw // 2, self.Npw // 2 + 1)
        return ni[:, None] - ni

    @property
    def hamMat(self):
        return np.fft.ifft(self.veff)[self._G_indices] + 0.5 * self._Gvec**2 * np.eye(self.Npw)

    def _get_fermi(self, x):
        x = np.atleast_1d(x)
        results = np.zeros_like(x)
        cond = (x / self.kT < 50)
        results[cond] = 1 / (1 + np.exp(x[cond] / self.kT))
        return results.squeeze()

    def _target_mu(self, mu):
        return np.sum(self._get_fermi(self.vals - mu)) - self.Nel

    @property
    def mu(self):
        if self._mu is None:
            self._mu, = fsolve(self._target_mu, x0=self.vals[self.Nel])
        return self._mu

    @property
    def vals(self):
        if self._vals is None:
            self._vals, self._vecs = np.linalg.eigh(self.hamMat)
        return self._vals

    @property
    def _vals_trunc(self):
        return self.vals[self.focc > self._min_focc]

    @property
    def vecs(self):
        if self._vecs is None:
            self._vals, self._vecs = np.linalg.eigh(self.hamMat)
        return self._vecs

    @property
    def _vecs_trunc(self):
        return self.vecs[:, self.focc > self._min_focc]

    @vals.setter
    def vals(self, v):
        self._mu = None
        self._vals = v

    @vecs.setter
    def vecs(self, v):
        self._vecs = v

    @property
    def focc(self):
        return self._get_fermi(self.vals - self.mu)

    @property
    def _focc_trunc(self):
        return self.focc[self.focc > self._min_focc]

    def get_rho(self):
        psi_expand = np.zeros(shape=(2 * self.Npw, len(self._focc_trunc))) * 1j
        psi_expand[0:self.Npw // 2] = self._vecs_trunc[0:self.Npw // 2]
        psi_expand[-self.Npw // 2:] = self._vecs_trunc[-self.Npw // 2:]
        psi = np.fft.fft(psi_expand, axis=0) / np.sqrt(self.L)
        return np.sum(np.absolute(psi)**2 * self._focc_trunc, axis=-1)

    @property
    def Ekin(self):
        return 0.5 * np.einsum(
            'j,ij,i->', self._focc_trunc, np.absolute(self._vecs_trunc)**2, self._Gvec**2,
            optimize=True
        )

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho_in):
        self._rho = np.array(rho_in)
        self._rho_G = None
        self._mu = None
        self._vals = None
        self._vecs = None

    @property
    def _rho_G(self):
        if self._rho_G is None:
            self._rho_G = np.fft.ifft(self.rho)
            self._rho_G += self.rhonuc_G
            self._rho_G[0] = 0.
        return self._rho_G

    @property
    def _alpha(self):
        return -3 / 4 * (3 / np.pi) ** (1 / 3)

    @property
    def eXc(self):
        return np.sum(self.rho[self.rho > 0]**(4. / 3.)) * self._alpha * self._dL / 2

    @property
    def vXc(self):
        return (4. / 3.) * self._alpha * np.maximum(self.rho, 0)**(1. / 3.)

    @property
    def V_G(self):
        return self._rho_G * self.coulomb

    @property
    def vH(self):
        return np.real(np.fft.fft(self.V_G))

    @property
    def veff(self):
        veff = self.vH + self.vXc
        if 'v_loc' in self.__dir__():
            veff += self.v_loc
        return veff

    @property
    def eLoc(self):
        if 'v_loc' in self.__dir__():
            return sum(self.v_loc * self.rho) * self._dL / 2
        else:
            return 0.

    @property
    def E_H(self):
        return 0.5 * np.vdot(self._rho_G, self.V_G).real * self.L

    def computePot(self, rho):
        self.rho = rho

    def getEnergy(self):
        return self.Ekin + self.eXc + self.E_H + self.eLoc
