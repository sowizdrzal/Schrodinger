
# class ConstsAndParams:
# This class take parameters from choose function and keep it with all necessary constants and parametes


class ConstsAndParams:
    def __init__(self):

        # parameters to verify function
        y = 'yes'
        n = 'no'
        text_sm = 'Do you want program to calculate schrodinger eq. for stable mass of electron and hole? Yes/no'
        text_nm = 'Do you want program to use Numerov method? Yes/no'
        text_ps = 'Do you want program to use poisson distribution to correct calculation? Yes/no'
        text_ll = 'Please type the left length of well potential (int - preferable 10)'
        text_rl = 'Please type the right length of well potential (int - preferable 10)'
        text_wl = 'Please type the well length of well potential (int - preferable 20)'
        text_s = 'Pleas type numerical step (float - preferable 0.05)'
        
        # choice of program options
        
        self.stable_mass = self.verify_str(y, n, text_sm)
        self.numerv = self.verify_str(y, n, text_nm)
        self.poisson = self.verify_str(y, n, text_ps)

        self.left_length = self.verify(text_ll, int())
        self.well_length = self.verify(text_wl, int())
        self.right_length = self.verify(text_rl, int())
        self.step = self.verify(text_s, float())

        # CONSTANTS
        self.h_bar = 6.582119514*10**-16
        self.epsilon = 8.85*10**-12
        self.m = 5.68*10**(-12)

        # SELF CONSISTENT
        self.self_consistent = 1
        self.self_parameter = 0.001
        self.iter = 5

        # GaAs eV
        self.gaas_gap = 1.424
        self.m_e_gaas = 0.063
        self.m_h_gaas = 0.51

        # InAs eV
        self.inas_gap = 0.354
        self.m_e_inas = 0.023
        self.m_h_inas = 0.41

        # VBO
        self.VBO = 0.5
        self.hole_barrier = self.VBO*(self.gaas_gap-self.inas_gap)
        self.electron_barrier = (1-self.VBO)*(self.gaas_gap-self.inas_gap)

    # verify:
    # This function takes inputs from the user, check it for format purpose
    # and return crucial parameters for numerical model

    def verify(self, warning, typeof):

        inp = None
        while type(inp) != type(typeof):
            print(warning)
            inp = input()
            try:
                inp = type(typeof)(inp)
            except ValueError:
                print('That is not a number!!!')

        return inp

    def verify_str(self, y, n, text):

        inp = ''
        variable = None

        while inp != y and inp != n:
            print(text)
            try:
                inp = input()
                if inp.lower() == y or inp.lower() == y[0]:
                    variable = True
                    break
                elif inp.lower() == n or inp.lower() == n[0]:
                    variable = False
                    break
                else:
                    print('Invalid answer!!!')
            except ValueError:
                print('Wrong answer!!!')

        return variable

