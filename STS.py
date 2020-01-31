import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import curve_fit
from scipy.optimize import brute
from scipy.optimize import Bounds



class STSSolution:
    safely_created = True  # Flag variable which alerts the user if the data is read in incorrectly.
    # Check for corrupted data or misaligned data if this value gets flipped

    path = None  # Path to data set from the current exec. env.
    data = []  # Init the array which holds the data |S21|
    dataWidth = 0  # Number of voltages swept
    dataLength = 0  # Number of frequencies swept
    freq_span = []  # Array containing the frequency axis
    volt_span = []  # Array containing the voltage axis
    voltage_offset = []  # Array containing the voltage offset axis
    delta_f = None  # Array containing the values of delta fr from min. frequencies
    minimum_frequencies = None  # Array containing the values at which |S21| is minimized for a given voltage
    correlation_function = None  # Array containing the self correlation function to extract period
    auto_correlation_matrix = None  # Array containing the loss function for correlating delta fr w/ square wave pulse
    phi_space = None  # Phi axis which ranges from -period/2 to period/2 (a.k.a. meander start)
    duty_space = None  # Duty cycle axis (0, 1)
    period = 0.0  # Extracted period (Pi param)
    phi_param = 0.0  # Extracted meander start (units of voltage)
    duty_param = 0.0  # Extracted duty cycle
    voltage_sweet_spot = 0.0  # Voltage location of sweet spot
    freq_sweet_spot = 0.0
    correlate_loc_max = None  # Index for correlation function. Should be swapped with a param option
    delta_fp = 0  # Range of frequencies in freq_span
    fc_param = 0
    interp_volt_axis = None
    interp_min_freq = None
    g_param = None
    fmax_ge_param = None
    d_param = None

    # TODO: Switch correlate_loc_max with a param

    def __init__(self, path):
        # To initialize a solution object, all thats needed is a .dat file like the one in examples
        # 5 step process to extracting parameters :::

        # 1. read in and organize the data
        # 2. autocorrelate the delta fr function against itself over the voltage offset axis
        # 3. Extract Period from step 2
        # 4. Using period from step 3 --> Autocorrelate against square wave / delta fr and minimize the loss function
        # 5. Plug into the formula for sweet spot

        self.path = path  # Initialize the path

        self.configure_data()
        self.correlate_period()
        self.extract_period()
        self.extract_duty_phi()
        self.extract_sweet_spot()

    def configure_data(self):

        self.data = np.loadtxt(self.path)

        self.dataLength = self.data.shape[0]
        self.dataWidth = self.data.shape[1]

        # with open(self.path) as f:  # Parse the .dat file and read in data
        #     for line in f:
        #         self.dataLength += 1
        #         self.data.append(np.array(line.split(), dtype=np.float))

        # self.dataWidth = len(self.data[0])

        self.check_data()  # Perform uniformity check

        # self.data = np.array(self.data)  # cast to numpy array

        # Gather the freq/volt information from the path variable. TODO: Add param to __init__ to make this optional

        m = re.search("fr(.+?)_(.+?)_Pr", self.path)  # Use regex to find the parameters of the experiment
        freq_range_min, freq_range_max = np.float(m.group(1)), np.float(
            m.group(2))  # Regex usage to extract the parameters
        n = re.search("_V(.+?)_(.+?)_", self.path)
        volt_range_min, volt_range_max = np.float(n.group(1)), np.float(n.group(2))  # More of the same

        self.freq_span = np.linspace(freq_range_min, freq_range_max, self.dataLength)
        self.delta_fp = self.freq_span[-1] - self.freq_span[0]
        self.volt_span = np.linspace(volt_range_min, volt_range_max, self.dataWidth)

        self.voltage_offset = np.array([self.volt_span[i] - self.volt_span[0] for i in range(self.dataWidth)])

        self.minimum_frequencies = np.zeros(self.dataWidth)  # Create array to iterate over

        for i in range(self.dataWidth):
            self.minimum_frequencies[i] = self.freq_span[
                np.argmin(self.data[:, i])]  # Find minimum of |S21| for each voltage

        # Compute delta fr
        self.delta_f = self.minimum_frequencies - (
            np.repeat(np.mean(self.minimum_frequencies), len(self.minimum_frequencies)))

    def check_data(self):

        for line in self.data:
            if len(line) != self.dataWidth:
                self.safely_created = False
                print("Warning, inconsistent data width")

        if self.safely_created:
            print("Data read in successfully. Continue processing")

    def correlate_period(self):
        self.correlation_function = np.zeros(self.dataWidth)  # Array for storing data
        for i in range(self.dataWidth):
            self.correlation_function[i] = (self.autocorrelation_function(self.delta_f, self.delta_f, i))

    def extract_period(self):

        loc_min = np.argmin(self.correlation_function)
        self.correlate_loc_max = np.argmax(self.correlation_function[loc_min:]) + loc_min

        self.period = self.voltage_offset[self.correlate_loc_max-1]

    def extract_duty_phi(self):

        self.auto_correlation_matrix = np.zeros(50 * 50).reshape(50, 50)
        self.phi_space = np.linspace(-self.period / 2.0, self.period / 2.0, 50)
        self.duty_space = np.linspace(0, .99, 50)
        for i, phi in enumerate(self.phi_space):
            for j, duty in enumerate(self.duty_space):
                r = self.autocorrelation_function(self.delta_f,
                                                  self.square_function(self.voltage_offset, phi, duty, self.period,
                                                                       max(self.delta_f)), 0)

                self.auto_correlation_matrix[i, j] = -1 * r

        #ind = np.unravel_index(np.argmin(self.auto_correlation_matrix, axis=None), self.auto_correlation_matrix.shape)
        #
        # row = ind[0]
        # col = ind[1]

        lin_index = np.argmin(self.auto_correlation_matrix)
        row = lin_index // 50
        col = lin_index % 50
        self.phi_param = self.phi_space[row]
        self.duty_param = self.duty_space[col]

    def extract_sweet_spot(self):

        self.voltage_sweet_spot = self.phi_param + self.period * self.duty_param / 2 - self.period

        self.period = self.voltage_offset[self.correlate_loc_max+1]
        voltage_index = 0
        for i, v in enumerate(self.volt_span):
            if self.voltage_sweet_spot > v:
                voltage_index = i
        self.freq_sweet_spot = self.freq_span[np.argmin(self.data[:, voltage_index])]

    def get_slice_freq_vs_amp(self, voltage):

        index = 0
        if voltage > self.volt_span[-1]:
            print("Voltage outside span. Using i = dataWidth - 1")
            index = self.dataWidth - 1
        elif voltage < self.volt_span[0]:
            print("Voltage outside span. Using i = 0")
        else:
            for i, val in enumerate(self.volt_span):
                if voltage >= val:
                    print("Found voltage in span. Taking LHS value")
                    index = i - 1

        return self.data[:, index]

    @staticmethod
    def autocorrelation_function(yn, yn_l, offset_index):

        ##Inputs :the delta functions, and the index l of the current offset
        total = 0

        for i, value in enumerate(yn):
            if i - offset_index >= 0:
                total += yn[i] * yn_l[i - offset_index]
        return total

    @staticmethod
    def square_function(current, phi, duty_cycle, period, amplitude):
        # Square function returns a rectangular pulse with pulse width (period * duty_cycle)
        return amplitude * signal.square(current * 2.0 * np.pi / period - phi, duty_cycle)

    def visualize_data(self):

        fig = plt.figure(figsize=(12, 6))

        extent = [self.volt_span[0], self.volt_span[-1], self.freq_span[0], self.freq_span[-1]]
        img = plt.imshow(self.data, cmap='gist_rainbow_r', origin='lower', aspect='auto', extent=extent)
        plt.xlabel(r"$V$")
        plt.ylabel(r'$f_p$ [GHz]')
        cbar = plt.colorbar(img)
        cbar.ax.set_ylabel(r"$|{S_{21}|$")

        return fig
    def visualize_comparison(self):
        fig, ax = plt.subplots(figsize=(12,12))
        if(self.fc_param):
            print("Hamiltonian parameters saved...\n \t Creating Comparison plot")
            ax.plot(self.interp_volt_axis, self.interp_min_freq, label='Original data (interpolated)')
            ax.plot(self.interp_volt_axis, self.f_function(self.interp_volt_axis,self.fc_param,
                                           self.g_param, self.fmax_ge_param, self.d_param, self.period, self.voltage_sweet_spot), label='Fit')
            plt.legend()
            plt.grid()
        else:
            print("Hamiltonian parameters not computed...\n Run curve_fit_STS or extract_hamiltonian_params")

        return fig, ax
    def visualize_autoCorrelation(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.voltage_offset, self.correlation_function)
        ax.scatter(self.voltage_offset[self.correlate_loc_max], self.correlation_function[self.correlate_loc_max])

        return fig, ax

    def visualize_duty_phi(self):
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))

        extent = [0, 1.0, self.phi_space[0], self.phi_space[-1]]
        img = axs[0].imshow(self.auto_correlation_matrix, origin='lower', aspect='auto', extent=extent, cmap='plasma')
        ax_div = make_axes_locatable(axs[0])
        # add an axes above the main axes.
        cax2 = ax_div.append_axes("top", size="7%", pad="2%")
        cb2 = colorbar(img, cax=cax2, orientation="horizontal")
        # change tick position to top. Tick position defaults to bottom and overlaps
        # the image.
        cax2.xaxis.set_ticks_position("top")
        cax2.set_xlabel(r"$\mathcal{L}$")
        axs[0].scatter(self.duty_param, self.phi_param, color='red', marker='X')
        axs[0].set_xlabel(r'Duty Cycle')
        axs[0].set_ylabel(r"Meander Start $(\Phi)$")

        axs[1].plot(self.voltage_offset, self.delta_f)
        axs[1].plot(self.voltage_offset, \
                    self.square_function(self.voltage_offset, self.phi_param, self.duty_param, self.period,
                                         max(self.delta_f)))

        return fig, axs

    def get_slice_at_V(self, voltage):
        fig, ax = plt.subplots(figsize=(12, 6))

        volt_loc = -1
        for i, value in enumerate(self.volt_span):
            if voltage >= value and voltage <= self.volt_span[i + 1 % self.dataWidth]:
                volt_loc = i
        if volt_loc == -1:
            print("Voltage not found in sweep bounds... Returning empty slice")
            return fig, ax

        ax.plot(self.freq_span, self.data[:, volt_loc])
        ax.set_xlabel(r"$f_p$ [GHz]")
        ax.set_ylabel(r"$|S_{21}|$ [a.u.]")

        return fig, ax

    def get_slice_at_F(self, freq):
        fig, ax = plt.subplots(figsize=(12, 6))

        freq_loc = -1
        for i, value in enumerate(self.freq_span):
            if freq >= value and freq <= self.freq_span[i + 1 % self.dataLength]:
                freq_loc = i

        if freq_loc == -1:
            print("Frequency not found in sweep bounds... Returning empty slice")
            return fig, ax
        ax.plot(self.volt_span, self.data[freq_loc, :])
        ax.set_xlabel(r"Voltage")
        ax.set_ylabel(r"$|S_{21}|$ [a.u.]")

        return fig, ax

    def get_period(self):
        return self.period

    def get_sweet_spot(self):
        return self.voltage_sweet_spot


    def f_ge_func(self, I_i, d, fmax_ge, period, voltage_sweet_spot):

        cos_term = np.cos(np.pi * (I_i - self.voltage_sweet_spot) / period) ** 2
        sin_term = d ** 2 * np.sin(np.pi * (I_i - self.voltage_sweet_spot) / period) ** 2
        sum_term = (cos_term + sin_term)
        sum_term = sum_term ** (1.0 / 4.0)

        return fmax_ge * sum_term

    def f_function(self, voltage_i, f_c, g, fmax_ge, d, period, voltage_sweet_spot):
        f_ge = self.f_ge_func(voltage_i, d, fmax_ge, period, voltage_sweet_spot)

        lhs = (f_c + f_ge) / 2.0
        sqrt_stuff = np.sqrt(g ** 2.0 + ((f_ge - f_c) ** 2) / 4.0)
        plus = lhs + sqrt_stuff

        return plus

    def m_function(self, voltage_i, f_c, g, fmax_ge, d, period, voltage_sweet_spot):
        return self.f_function(voltage_i, f_c, g, fmax_ge, d, period, voltage_sweet_spot)


        # if abs(f_plus - f_c) < self.delta_fp / 2:
        #     return f_plus
        # else:
        #     return f_minus

    @staticmethod
    def loss_function(x, self):  # x := ([f_c, g, fmax_ge, d]):
        total = 0

        f_c = x[0]
        g = x[1]
        fmax_ge = x[2]
        d = x[3]
        period = x[4]
        voltage_sweet_spot = x[5]
        for i, v_i in enumerate(self.interp_volt_axis):
            total += (self.interp_min_freq[i] - self.m_function(v_i, f_c, g, fmax_ge, d, period, voltage_sweet_spot)) ** 2
        return total

    def extract_hamiltonian_params(self):

        self.interp_volt_axis = np.linspace(self.volt_span[0], self.volt_span[-1], 100)

        interpolation_scheme = interp1d(self.volt_span, self.minimum_frequencies, kind='quadratic')
        self.interp_min_freq = interpolation_scheme(self.interp_volt_axis)

        bounds_array_upper = (np.mean(self.minimum_frequencies)+.001, .1, 12.0, 0.9, self.volt_span[-1], self.volt_span[-1])
        bounds_array_lower = (np.mean(self.minimum_frequencies)- .001, .09, 4.0, 0.0, self.volt_span[0], self.volt_span[0])

        x_initial = [np.mean(self.minimum_frequencies)-.0005, .09, np.max(self.minimum_frequencies), 0.5, self.period,
                                                                self.voltage_sweet_spot]
        bounds = ParamBounds(xmax=bounds_array_upper, xmin=bounds_array_lower)

        print("Attempting to minimize loss function for Hamiltonian parameters")
        minimizer_kwargs = {'method':'L-BFGS-B', "args": self}
        basin_result = basinhopping(self.loss_function, x_initial, niter = 200, accept_test = bounds,
                                    minimizer_kwargs=minimizer_kwargs ,T=1e-5, stepsize=0.001)



        if(basin_result.pop('lowest_optimization_result')['success']):
            print("Loss function minimized successfully\n\t Saving hamiltonian params to solution object")
            self.fc_param = basin_result.x[0]
            self.g_param = basin_result.x[1]
            self.fmax_ge_param = basin_result.x[2]
            self.d_param = basin_result.x[3]
            self.period = basin_result.x[4]
            self.voltage_sweet_spot = basin_result.x[5]
        else:
            print("Loss function minimization with basin hopping failled\n\t Attempting nelder-mead minimization")
            self.extract_nelder_mead(x_initial, bounds)

    def curve_fit_STS(self):

        print("Attempting a curve fit solution to extract hamiltonian parameters")
        #f_c, g, fmax_ge, d, p, Iss
        p0 = [self.fc_param, self.g_param, self.fmax_ge_param, self.d_param, self.period, self.voltage_sweet_spot]
        curve_fit_params = curve_fit(self.f_function, self.interp_volt_axis, self.interp_min_freq, p0=p0)
        self.fc_param = curve_fit_params[0][0]
        self.g_param = curve_fit_params[0][1]
        self.fmax_ge_param = curve_fit_params[0][2]
        self.d_param = curve_fit_params[0][3]
        self.period = curve_fit_params[0][4]
        self.voltage_sweet_spot = curve_fit_params[0][5]


    def extract_nelder_mead(self, x_initial=[],bounds=[]):
        if (x_initial==[]):
            print("No initial values passed for x_inital, using mean values")
            x_initial = np.array([np.mean(self.minimum_frequencies)-.0005, .03, 6.0, .5])

        if (bounds==[]):
            print("No initial bounds passed, using assumed bounds")
            lb = bounds_array_lower = (np.mean(self.minimum_frequencies)- .001, .09, 4.0, 0.0)
            ub = (np.mean(self.minimum_frequencies)+.001, .1, 12.0, 0.9)
            bounds = Bounds(lb, ub)

        mead_result = minimize(self.loss_function, x_initial,method='SLSQP', args=(self),
                               bounds=bounds, callback=print_mead)

        # if(np.all(mead_result.x <= bounds.xmax)) and (np.all(mead_result.x >= bounds.xmin)):
        print("Found solution using nelder-mead minimization algorithm")
        print("\tSaving hamiltonian params to solution object")
        self.fc_param = mead_result.x[0]
        self.g_param = mead_result.x[1]
        self.fmax_ge_param = mead_result.x[2]
        self.d_param = mead_result.x[3]
        # else:
        #     if(mead_result.success):
        #         print("Successfully minimized function outside of bounds. Use caution proceeding")
        #         print("\t Saving hamiltonian params to solution object")
        #         self.fc_param = mead_result.x[0]
        #         self.g_param = mead_result.x[1]
        #         self.fmax_ge_param = mead_result.x[2]
        #         self.d_param = mead_result.x[3]
        #     else:
        #         print("Function not minimized. Parameters could not be extracted")



class ParamBounds(object):

    def __init__(self, xmax=(1.1, 1.1, 1.1, 1.1), xmin=(-1.1, -1.1, -1.1, -1.1)):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]

        # for i,y in enumerate(self.xmax):
        #     tmax = tmax and (x[i] < y)
        #     tmin = tmin and (self.xmin[i] > x[i])
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        return tmax and tmin

def print_fun(x, f, accepted):
    print("at minimum %.14f accepted %d" % (f, int(accepted)))

def print_mead(x):
    print("at minimum loc [%.14f, %.14f, %.14f, %.14f]" % (x[0], x[1], x[2], x[3]))