import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import brute

# /Users/aarontrowbridge/LabStuff/Single-Tone-Spectroscopy/AB40_S21vsFvsV_fr6.03_6.05_Pr-80_V-1.5_1.5_0.1_T0.084_Cav2_143704_mag.dat

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
    correlate_loc_max = None  # Index for correlation function. Should be swapped with a param option

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

        self.period = self.voltage_offset[self.correlate_loc_max]

    def extract_duty_phi(self):

        self.auto_correlation_matrix = np.zeros(50 * 50).reshape(50, 50)
        self.phi_space = np.linspace(-self.period / 2, self.period / 2, 50)
        self.duty_space = np.linspace(0, .99, 50)
        for i, phi in enumerate(self.phi_space):
            for j, duty in enumerate(self.duty_space):
                r = self.autocorrelation_function(self.delta_f,
                                                  self.square_function(self.voltage_offset, phi, duty, self.period,
                                                                       max(self.delta_f)), 0)

                self.auto_correlation_matrix[i, j] = -1 * r

        lin_index = np.argmin(self.auto_correlation_matrix)
        row = lin_index // 50
        col = lin_index % 50

        self.phi_param = self.phi_space[row]
        self.duty_param = self.duty_space[col]

    def extract_sweet_spot(self):
        self.voltage_sweet_spot = self.phi_param + self.period * self.duty_param / 2 - self.period

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

        fig, ax = plt.subplots(figsize=(12, 6))

        extent = [self.volt_span[0], self.volt_span[-1], self.freq_span[0], self.freq_span[-1]]
        img = ax.imshow(self.data, cmap='gist_rainbow_r', origin='lower', aspect='auto', extent=extent)
        ax.set_xlabel(r"$V$")
        ax.set_ylabel(r'$f_p$ [GHz]')
        cbar = plt.colorbar(img)
        cbar.ax.set_ylabel(r"$|{S_{21}|$")

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
