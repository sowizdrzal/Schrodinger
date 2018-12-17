import inputs
import numpy as np
import datetime
from scipy import linalg as LA
from scipy import integrate
import matplotlib.pyplot as plt
import inputs


class Onn:

    def __init__(self):

        self.consts = inputs.ConstsAndParams()
        # calculation of numerical space of potential
        self.n_left = int(self.consts.left_length / self.consts.step)
        self.n_right = int(self.consts.right_length / self.consts.step)
        self.n_well = int(self.consts.well_length / self.consts.step)
        self.n = self.n_left + self.n_well + self.n_right
        self.dx = self.consts.step * 10 ** (-9)
        self.constant = (-self.consts.h_bar ** 2) / (2 * self.dx ** 2)
        # zero potential matrixes for electron and hole
        self.v_oe = np.zeros((self.n, self.n))
        self.v_oh = np.zeros((self.n, self.n))

        # zero potential matrixes for electron and hole

        self.v_e = np.zeros((self.n, self.n))
        self.v_h = np.zeros((self.n, self.n))
        self.de = np.zeros((self.n, self.n))
        self.de_sm = np.zeros((self.n, self.n))
        self.dh = np.zeros((self.n, self.n))
        self.dh_sm = np.zeros((self.n, self.n))
        self.energies_e = []
        self.vectors_e = 0
        self.energies_h = []
        self.vectors_h = 0
        self.energies_e_sm = 0
        self.energies_h_sm = 0
        self.vectors_e_sm = 0
        self.vectors_h_sm = 0
        self.v_up_e = np.zeros((self.n, self.n))
        self.v_up_e_sm = np.zeros((self.n, self.n))
        self.v_up_h = np.zeros((self.n, self.n))
        self.choice_E_e = 0
        self.choice_F_e_sm = 0
        self.choice_E_e_sm = 0
        self.choice_E_h = 0
        self.choice_F_e = 0
        self.choice_F_h = 0
        self.he_sm = 0
        self.hh_sm = 0
        self.he = 0
        self.hh = 0

        self.time = datetime.datetime.now()
        self.file_name = 'schrodinger_out'
        

    def _parameters(self):

        time = datetime.datetime.now()
        str_n = ''
        if self.consts.numerv and self.consts.poisson:
            str_n = 'Calculations were provide with Numerov method, Poisson distribution and self-consistent pseudopotentials. '
        elif self.consts.numerv == False and self.consts.poisson:
            str_n = 'Calculations were provide with Poisson distribution and self-consistent pseudopotentials.'
        elif self.consts.numerv == False and self.consts.poisson == False:
            str_n = 'Calculations were provide without Numerov method, Poisson distribution and self-consistent pseudopotentials.\n '
        else:
            str_n = 'Calculations were provide with Numerov method.'

        if self.consts.stable_mass:
            str_n += ' Program calculate values \n for variable boundary mass and stable mass for comparison.'
        else:
            str_n += ' Program calculate values for variable boundary mass.'

        self.file_name += f'{time.strftime("%y%m%d_%H:%M")}.txt'

        with open(self.file_name, 'w') as filehandle:
            filehandle.write(f'START OF CALCULATIONS: {self.time}\n\n\nNUMERICAL CALCULATION OF SCHRÖDINGER EQUATION\nfor electron and hole in finite square potential well '
                             f'for InAs/GaAs quantum dots\n{str_n}\n************* STARTUP PARAMETERS ************\n left length: {self.consts.left_length}[nm]'
                             f'\n right length: {self.consts.right_length}[nm]\n well length: {self.consts.well_length}[nm]'
                             f'\n numerical step: {self.consts.step}\ndx: {self.dx}\n******************** GaAs ********************\n'
                             f'effective mass of electron {self.consts.m_e_gaas} [eV]\neffective mass of hole: {self.consts.m_h_gaas} [eV]\ngap: {self.consts.gaas_gap} [eV]\n'
                             f'n******************** InAs *******************\neffective mass of electron: {self.consts.m_e_inas} [eV]\neffective mass of hole: {self.consts.m_h_inas}'
                             f' [eV]\ngap: {self.consts.inas_gap} [eV]\n**********************************************\n'
                             f'Valence–Bond Order (VBO): {self.consts.VBO}\nelectron barrier: {self.consts.electron_barrier} [eV]\nhole barrier: {self.consts.hole_barrier} [eV]'
                             f'\n**********************************************\n')

            filehandle.write(
                f'\n HAMILTONIAN MATRIX FOR ELECTRON SIZE:{self.he.shape}\n')
            filehandle.write(
                f'\n HAMILTONIAN MATRIX FOR HOLE SIZE:{self.he.shape}\n')

            filehandle.write('\n\nENERGIES  FOR (InAs/GaAs QD) ELECTRON\n\n')
            c = 0
            for item in self.energies_e:
                c += 1
                filehandle.write(f'{c} {item} [eV]\n')

            filehandle.write(
                '\n\nWAVEFUNCTIONS FOR (InAs/GaAs QD) ELECTRON\n\n')
            for i in range(len(self.energies_e)):
                filehandle.write(f'   \u03C8 {i}      ')

            filehandle.write('\n')
            for item in self.choice_F_e:
                filehandle.write(f'{item}\n')

            filehandle.write('\n\nENERGIES  FOR (InAs/GaAs QD) HOLE\n\n')
            c = 0
            for item in self.energies_h:
                c += 1
                filehandle.write(f'{c} {item} [eV]\n')

            filehandle.write('\n\nWAVEFUNCTIONS FOR (InAs/GaAs QD) HOLE\n\n')
            for i in range(len(self.energies_h)):
                filehandle.write(f'    \u03C8 {i}     ')

            filehandle.write('\n')
            for item in self.choice_F_h:
                filehandle.write(f'{item}\n')

            filehandle.write(f'\n**********************************************\n'
                             f'\nPOTENTIAL IN NUMERCIAL SPACE\n\n\n{self.v_up_e}')

            if self.consts.stable_mass:
                filehandle.write('\n\nFOR STABLE MASS\n\n')
                filehandle.write(
                    '\n\nENERGIES  FOR (InAs/GaAs QD) ELECTRON\n\n')
                c = 0
                for item in self.energies_e_sm:
                    c += 1
                    filehandle.write(f'{c} {item} [eV]\n')

                filehandle.write(
                    '\n\nWAVEFUNCTIONS FOR (InAs/GaAs QD) ELECTRON\n\n')
                for i in range(len(self.energies_e_sm)):
                    filehandle.write(f'   \u03C8 {i}      ')

                filehandle.write('\n')
                for item in self.choice_F_e_sm:
                    filehandle.write(f'{item}\n')

    def _mass(self):

        # calcuation of effective mass for gaas and inas
        m_e_gaas = self.consts.m_e_gaas * self.consts.m
        m_h_gaas = self.consts.m_h_gaas * self.consts.m

        m_e_inas = self.consts.m_e_inas * self.consts.m
        m_h_inas = self.consts.m_h_inas * self.consts.m

        for i in range(0, self.n_left):
            self.v_oe[i, i] = m_e_gaas
            self.v_oh[i, i] = m_h_gaas
        for i in range(self.n_left, self.n_left + self.n_well):
            self.v_oe[i, i] = m_e_inas
            self.v_oh[i, i] = m_h_inas
        for i in range(
                self.n_left +
                self.n_well,
                self.n_well +
                self.n_left +
                self.n_right):
            self.v_oe[i, i] = m_e_gaas
            self.v_oh[i, i] = m_h_gaas

        return self.v_oe, self.v_oh

    def _e_potential(self):

        # calculation of potential on well boundaries
        for i in range(0, self.n_left):
            self.v_e[i, i] = self.consts.electron_barrier
            self.v_h[i, i] = self.consts.hole_barrier

        for i in range(
                self.n_left +
                self.n_well,
                self.n_well +
                self.n_right +
                self.n_left):
            self.v_e[i, i] = self.consts.electron_barrier
            self.v_h[i, i] = self.consts.hole_barrier

        # If we want to use Numerov method, we have to calculate tree-diagonal
        # matrix for next steps
        #  x y 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  y x y 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  0 x y x 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  0 0 x y x 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  0 0 0 x y x 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  0 0 0 0 x y x 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  ......
        # x = 5/6 * barrier
        # y = 1/12 * barrier
        if self.consts.numerv:

            for i in range(0, self.n_left):
                self.v_e[i, i] = 5 / 6 * self.consts.electron_barrier
                self.v_h[i, i] = 5 / 6 * self.consts.hole_barrier
                if i > 0:
                    self.v_e[i - 1, i] = 1 / 12 * self.consts.electron_barrier
                    self.v_e[i, i - 1] = 1 / 12 * self.consts.electron_barrier

                    self.v_h[i - 1, i] = 1 / 12 * self.consts.hole_barrier
                    self.v_h[i, i - 1] = 1 / 12 * self.consts.hole_barrier

            for i in range(
                    self.n_left +
                    self.n_well,
                    self.n_well +
                    self.n_right +
                    self.n_left):
                self.v_e[i, i] = 5 / 6 * self.consts.electron_barrier
                self.v_h[i, i] = 5 / 6 * self.consts.hole_barrier
                if i > 0:
                    self.v_e[i - 1, i] = 1 / 12 * self.consts.electron_barrier
                    self.v_e[i, i - 1] = 1 / 12 * self.consts.electron_barrier

                    self.v_h[i - 1, i] = 1 / 12 * self.consts.hole_barrier
                    self.v_h[i, i - 1] = 1 / 12 * self.consts.hole_barrier

        return self.v_e, self.v_h

    # This function is used to calculations of hamiltoniam matrices
    # for two cases with variable mass and with stable mass,
    # in first step function calculated mass from previous _mass function.
    # in next steps function check if size of hamiltonian matrices
    # have the same size as potential energy matrix, if they are program
    # starts calculated values for 3 diagonals (i-1, i, i+1)

    def hamiltonian(self):

        self._e_potential()
        me, mh = self._mass()

        if self.de.size == self.v_e.size:
            for i in range(1, self.n - 1):
                self.de[i, i] = -0.5 * self.constant * \
                    (1 / me[i - 1, i - 1] + 2 /
                     me[i, i] + 1 / me[i + 1, i + 1])
                self.de[i, i + 1] = 0.5 * self.constant * (
                    1 / me[i + 1, i + 1] + 1 / me[i, i])
                self.de[i, i - 1] = 0.5 * self.constant * (
                    1 / me[i - 1, i - 1] + 1 / me[i, i])

            self.de[0, 0] = -0.5 * self.constant * (2 / me[0,
                                                           0] + 1 / me[1, 1])
            self.de[0, 1] = 0.5 * self.constant * (1 / me[0, 0] + 1 / me[0, 0])
            self.de[self.n - 1, self.n - 1] = -0.5 * self.constant * \
                (1 / me[self.n - 2, self.n - 2] +
                 2 / me[self.n - 1, self.n - 1])
            self.de[self.n - 1, self.n - 2] = 0.5 * self.constant * \
                (1 / me[self.n - 1, self.n - 1] +
                 1 / me[self.n - 2, self.n - 2])
        else:
            print('Hamiltonian have wrong size')
        self.he = self.de + self.v_e

        if self.dh.size == self.v_e.size:
            for i in range(1, self.n - 1):
                self.dh[i, i] = -0.5 * self.constant * \
                    (1 / mh[i - 1, i - 1] + 2 /
                     mh[i, i] + 1 / mh[i + 1, i + 1])
                self.dh[i, i + 1] = 0.5 * self.constant * (
                    1 / mh[i + 1, i + 1] + 1 / mh[i, i])
                self.dh[i, i - 1] = 0.5 * self.constant * (
                    1 / mh[i - 1, i - 1] + 1 / mh[i, i])

            self.dh[0, 0] = -0.5 * self.constant * (2 / mh[0,
                                                           0] + 1 / mh[1, 1])
            self.dh[0, 1] = 0.5 * self.constant * (1 / mh[0, 0] + 1 / mh[0, 0])
            self.dh[self.n - 1, self.n - 1] = -0.5 * self.constant * \
                (1 / mh[self.n - 2, self.n - 2] +
                 2 / mh[self.n - 1, self.n - 1])
            self.dh[self.n - 1, self.n - 2] = 0.5 * self.constant * \
                (1 / mh[self.n - 1, self.n - 1] +
                 1 / mh[self.n - 2, self.n - 2])
        else:
            print('Hamiltonian have wrong size')
        self.hh = self.dh + self.v_h

        if self.consts.stable_mass:

            m_e_inas = self.consts.m_e_inas * self.consts.m
            m_h_inas = self.consts.m_h_inas * self.consts.m

            if self.de.size == self.v_e.size:
                for i in range(1, self.n - 1):
                    self.de_sm[i, i] = -0.5 * self.constant * \
                        (1 / m_e_inas + 2 / m_e_inas + 1 / m_e_inas)
                    self.de_sm[i, i + 1] = 0.5 * self.constant * \
                        (1 / m_e_inas + 1 / m_e_inas)
                    self.de_sm[i, i - 1] = 0.5 * self.constant * \
                        (1 / m_e_inas + 1 / m_e_inas)

                self.de_sm[0, 0] = -0.5 * self.constant * \
                    (2 / m_e_inas + 1 / m_e_inas)
                self.de_sm[0, 1] = 0.5 * self.constant * \
                    (1 / m_e_inas + 1 / m_e_inas)
                self.de_sm[self.n - 1, self.n - 1] = -0.5 * \
                    self.constant * (1 / m_e_inas + 2 / m_e_inas)
                self.de_sm[self.n - 1, self.n - 2] = 0.5 * \
                    self.constant * (1 / m_e_inas + 1 / m_e_inas)
            else:
                print('Hamiltonian have wrong size')
            self.he_sm = self.de_sm + self.v_e

            if self.dh.size == self.v_e.size:
                for i in range(1, self.n - 1):
                    self.dh_sm[i, i] = -0.5 * self.constant * \
                        (1 / m_h_inas + 2 / m_h_inas + 1 / m_h_inas)
                    self.dh_sm[i, i + 1] = 0.5 * self.constant * \
                        (1 / m_h_inas + 1 / m_h_inas)
                    self.dh_sm[i, i - 1] = 0.5 * self.constant * \
                        (1 / m_h_inas + 1 / m_h_inas)

                self.dh_sm[0, 0] = -0.5 * self.constant * \
                    (2 / m_h_inas + 1 / m_h_inas)
                self.dh_sm[0, 1] = 0.5 * self.constant * \
                    (1 / m_h_inas + 1 / m_h_inas)
                self.dh_sm[self.n - 1, self.n - 1] = -0.5 * \
                    self.constant * (1 / m_h_inas + 2 / m_h_inas)
                self.dh_sm[self.n - 1, self.n - 2] = 0.5 * \
                    self.constant * (1 / m_h_inas + 1 / m_h_inas)

            self.hh_sm = self.dh_sm + self.v_h

        return self.he, self.hh, self.he_sm, self.hh_sm

    def _nv(self, consts):
        
        nv = np.zeros((self.n, self.n))
        for i in range(self.n):
            nv[i, i] = 5 / 6
            if i > 1:
                nv[i, i - 1] = 1 / 12
                nv[i - 1, i] = 1 / 12

        return nv

    # calculations of eigenvalues and eigenvectors

    def ev(self):

        # creating numerov matrix
        nv = np.zeros((self.n, self.n))
        for i in range(self.n):
            nv[i, i] = 5 / 6
            if i > 1:
                nv[i, i - 1] = 1 / 12
                nv[i - 1, i] = 1 / 12

        if (self.consts.numerv == False) and (
                self.consts.stable_mass == False):
            self.energies_e, self.vectors_e = LA.eigh(self.he)
            self.energies_h, self.vectors_h = LA.eigh(self.hh)

        elif (self.consts.numerv == False) and (self.consts.stable_mass == True):
            self.energies_e, self.vectors_e = LA.eigh(self.he)
            self.energies_h, self.vectors_h = LA.eigh(self.hh)
            self.energies_e_sm, self.vectors_e_sm = LA.eigh(self.he_sm)
            self.energies_h_sm, self.vectors_h_sm = LA.eigh(self.hh_sm)

        elif (self.consts.numerv) and (self.consts.stable_mass):
            self.energies_e, self.vectors_e = LA.eigh(self.he, nv)
            self.energies_h, self.vectors_h = LA.eigh(self.hh, nv)
            self.energies_e_sm, self.vectors_e_sm = LA.eigh(self.he_sm, nv)
            self.energies_h_sm, self.vectors_h_sm = LA.eigh(self.hh_sm, nv)
        else:
            self.energies_e, self.vectors_e = LA.eigh(self.he, nv)
            self.energies_h, self.vectors_h = LA.eigh(self.hh, nv)

        return self.energies_e, self.vectors_e, self.energies_h, self.vectors_h, self.energies_e_sm, self.energies_h_sm

    def _poisson_method(self, ro_e, ro_h):

        dx = self.consts.step * 10 ** (-9)
        second_derivative_e = np.zeros((len(ro_e), len(ro_e)))
        for i in range(len(ro_e)):
            second_derivative_e[i, i] = -2
            if i > 0:
                second_derivative_e[i - 1, i] = 1
                second_derivative_e[i, i - 1] = 1

        second_derivative_e_dx = second_derivative_e * (1 / dx)
        phi_e = LA.solve(second_derivative_e_dx,
                         (-(-1.6 * (10 ** (-19)) * ro_e))) / self.consts.epsilon

        second_derivative_h = np.zeros((len(ro_h), len(ro_h)))
        for i in range(len(ro_h)):
            second_derivative_h[i, i] = -2
            if i > 0:
                second_derivative_h[i - 1, i] = 1
                second_derivative_h[i, i - 1] = 1

        second_derivative_h_dx = second_derivative_h * (1 / dx)
        phi_h = LA.solve(second_derivative_h_dx,
                         (-(1.6 * (10 ** (-19)) * ro_h))) / self.consts.epsilon

        return phi_e, phi_h

    def self_consistent_method(self):

        if self.consts.poisson:
            pe, ph = self._poisson_method(
                np.abs(self.energies_e ** 2), np.abs(self.energies_h ** 2))
            error = self.consts.self_consistent
            count = 0
            while error > 0 or count < self.consts.iter:
                self.v_e = self.v_e + pe
                self.v_h = self.v_h + ph
                self.he, self.hh, he_sm, hh_sm = self.hamiltonian()
                energies_e, vectors_e, energies_h, vectors_h, energies_e_sm, energies_h_sm = self.ev()
                pe_1, ph_1 = self._poisson_method(
                    np.abs(energies_e ** 2), np.abs(energies_h ** 2))
                pe = self.consts.self_parameter * pe + \
                    (1 - self.consts.self_parameter) * pe_1
                ph = self.consts.self_parameter * ph + \
                    (1 - self.consts.self_parameter) * ph_1
                error = abs(energies_h[1] - self.energies_h[1])
                self.energies_e = energies_e
                self.energies_h = energies_h
                self.vectors_e = vectors_e
                self.vectors_h = vectors_h

                count += 1

        return self.energies_e, self.energies_h, self.vectors_e, self.vectors_h

    # This function choose which wave functions will fill in potential well by

    def function_choice(self):

        ran = 0.0
        ranges = 0
        self.v_up_e = np.diag(self.v_e) + (self.consts.inas_gap / 2)
        self.v_up_h = np.diag(self.v_h) * (-1) - (self.consts.inas_gap / 2)

        for i in range(len(self.energies_e)):
            if self.consts.electron_barrier > ran:
                ran = self.energies_e[i]
                ranges = i
            else:
                break

        self.choice_F_e = np.zeros(self.vectors_e[:, :ranges].shape)
        self.choice_E_e = np.zeros(ranges)
        L2_e = np.zeros(ranges)

        self.choice_F_h = np.zeros(self.vectors_h[:, :ranges].shape)
        self.choice_E_h = np.zeros(ranges)
        L2_h = np.zeros(ranges)

        x_e = len(self.vectors_e[:, 0])
        dx_e = x_e / self.n

        x_h = len(self.vectors_h[:, 0])
        dx_h = x_h / self.n

        # L2 normalization

        for i in range(ranges):
            L2_e[i] = (1 / integrate.simps(
                self.vectors_e[:, i] ** 2, dx=dx_e)) ** (1 / 2)
            self.choice_F_e[:, i] = L2_e[i] * self.vectors_e[:, i]
            L2_h[i] = (1 / integrate.simps(
                self.vectors_h[:, i] ** 2, dx=dx_h)) ** (1 / 2)
            self.choice_F_h[:, i] = L2_h[i] * self.vectors_h[:, i]

            self.choice_E_e[i] = self.energies_e[i]
            self.choice_E_h[i] = self.energies_h[i]

        self.choice_F_e = self.choice_F_e + (self.consts.inas_gap / 2)
        self.choice_F_h = self.choice_F_h - (self.consts.inas_gap / 2)

        # placing wave functions on energy levesl

        for i in range(ranges):
            self.choice_F_e[:, i] = self.choice_F_e[:, i] + self.choice_E_e[i]
            self.choice_F_h[:, i] = self.choice_F_h[:, i] - self.choice_E_h[i]

        # L2 normalization for stable mass

        if self.consts.stable_mass:
            self.v_up_e_sm = np.diag(self.v_e) + (self.consts.inas_gap / 2)
            self.choice_F_e_sm = np.zeros(self.vectors_e_sm[:, :ranges].shape)
            self.choice_E_e_sm = np.zeros(ranges)
            L2_e_sm = np.zeros(ranges)

            x_e_sm = len(self.vectors_e_sm[:, 0])
            dx_e_sm = x_e_sm / self.n

            for i in range(ranges):
                L2_e_sm[i] = (1 / integrate.simps(
                    self.vectors_e_sm[:, i] ** 2, dx=dx_e_sm)) ** (1 / 2)
                self.choice_F_e_sm[:, i] = L2_e_sm[i] * self.vectors_e_sm[:, i]

                self.choice_E_e_sm[i] = self.energies_e_sm[i]

            self.choice_F_e_sm = self.choice_F_e_sm + \
                (self.consts.inas_gap / 2)

            for i in range(ranges):
                self.choice_F_e_sm[:,
                                   i] = self.choice_F_e_sm[:,
                                                           i] + self.choice_E_e_sm[i]

        # Double check after the correction: selecting functions that fit into
        # the potential well

        f_e_boundary = self.choice_F_e[1, -1]
        v_max = max(self.v_up_e)

        while f_e_boundary > v_max:
            self.choice_F_e = self.choice_F_e[:, :-1]
            self.choice_F_h = self.choice_F_h[:, :-1]
            self.choice_F_e_sm = self.choice_F_e_sm[:, :-1]
            f_e_boundary = self.choice_F_e[1, -1]
            if f_e_boundary >= v_max:
                break

        n_ene = len(self.choice_F_e[0, :])
        self.energies_e = self.energies_e[:n_ene]
        self.energies_h = self.energies_h[:n_ene]
        if self.consts.stable_mass:
            self.energies_e_sm = self.energies_e_sm[:n_ene]

        self._parameters()

        return self.choice_F_e, self.choice_F_h, self.v_up_e, self.v_up_h, self.choice_F_e_sm,\
            self.v_up_e_sm, self.energies_e, self.energies_h, self.energies_e_sm

    def plotting(self):

        n = (
            self.consts.left_length /
            self.consts.step +
            self.consts.right_length /
            self.consts.step +
            self.consts.well_length /
            self.consts.step)
        x = np.linspace(0, self.consts.step * n, n)

        if self.consts.stable_mass:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(x, self.v_up_e_sm)
            plt.plot(x, self.choice_F_e_sm)
            plt.title('elektron (stała masa)')
            plt.xlabel('x[nm]')
            plt.ylabel('E [eV]')

            plt.subplot(2, 2, 2)
            plt.plot(x, self.v_up_e)
            plt.plot(x, self.choice_F_e)
            if self.consts.numerv and self.consts.poisson == False:
                plt.title('elektron (zmienna masa) + numerov')
            elif self.consts.numerv and self.consts.poisson:
                plt.title(('elektron samo-uzgodnienie'))
            else:
                plt.title('elektron (zmienna masa)')
            plt.xlabel('x[nm]')
            plt.ylabel('E [eV]')

            plt.subplot(2, 2, 4)
            plt.plot(x, self.v_up_h)
            plt.plot(x, self.choice_F_h)
            if self.consts.numerv and self.consts.poisson == False:
                plt.title('dziura (zmienna masa) + numerov')
            elif self.consts.numerv and self.consts.poisson:
                plt.title(('dziura samo-uzgodnienie'))
            else:
                plt.title('dziura (zmienna masa)')
            plt.xlabel('x[nm]')
            plt.ylabel('E [eV]')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            self.file_name = self.file_name[:-3]
            plt.savefig(f'{self.file_name}png')

        else:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(x, self.v_up_e)
            plt.plot(x, self.choice_F_e)
            if self.consts.numerv and self.consts.poisson == False:
                plt.title('elektron (zmienna masa) + numerov')
            elif self.consts.numerv and self.consts.poisson:
                plt.title(('elektron samo-uzgodnienie'))
            else:
                plt.title('elektron (zmienna masa)')
            plt.xlabel('x[nm]')
            plt.ylabel('E [eV]')

            plt.subplot(2, 2, 2)
            plt.plot(x, self.v_up_h)
            plt.plot(x, self.choice_F_h)
            if self.consts.numerv and self.consts.poisson == False:
                plt.title('dziura (zmienna masa) + numerov')
            elif self.consts.numerv and self.consts.poisson:
                plt.title(('dziura samo-uzgodnienie'))
            else:
                plt.title('dziura (zmienna masa)')
            plt.xlabel('x[nm]')
            plt.ylabel('E [eV]')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            self.file_name = self.file_name[:-3]
            plt.savefig(f'{self.file_name}png')

        plt.show()
