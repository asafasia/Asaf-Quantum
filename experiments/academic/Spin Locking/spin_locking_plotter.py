from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from helper import plotter
from helper import experiment_results
from helper.utility_functions import correct_axis, cos_wave_exp, exp_decay

# %% choose data
date = '22_12_2023'

exp_name = 'spin_locking_long_runnnnnnnn'

n = 1

data = experiment_results.ExperimentData.load_data(date, exp_name, n)

qubit = data['meta_data']["experiment_properties"]['qubit']
qubit_parameters = data['meta_data']["experiment_properties"]['qubit_parameters']

t = np.array(data['data']['t'])
f = np.array(data['data']['f'])
x = np.array(data['data']['x_data_r'])
y = np.array(data['data']['y_data_r'])
z = np.array(data['data']['z_data_r'])

print(y.shape)

r = np.sqrt((2 * x - 1) ** 2 + (2 * y - 1) ** 2 + (2 * z - 1) ** 2)

# %% 3d plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

t_mesh, d_mesh = np.meshgrid(t, f)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

im1 = axs[0, 0].pcolormesh(t_mesh * 1e6, d_mesh * 1e-6, x)
axs[0, 0].set_title('X basis')
axs[0, 0].set_ylabel('Detuning (µm)')  # Add y-axis label
divider1 = make_axes_locatable(axs[0, 0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1)

# Plot 2
im2 = axs[0, 1].pcolormesh(t_mesh * 1e6, d_mesh * 1e-6, y)
axs[0, 1].set_title('Y basis')
# axs[0, 1].set_ylabel('Detuning (µm)')
divider2 = make_axes_locatable(axs[0, 1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax2)

# Plot 3
im3 = axs[1, 0].pcolormesh(t_mesh * 1e6, d_mesh * 1e-6, z)
axs[1, 0].set_title('Z basis')
axs[1, 0].set_xlabel('Time (µs)')
axs[1, 0].set_ylabel('Detuning (µm)')
divider3 = make_axes_locatable(axs[1, 0])
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax3)

# Plot 4
im4 = axs[1, 1].pcolormesh(t_mesh * 1e6, d_mesh * 1e-6, r)
axs[1, 1].set_title('Bloch radius')
axs[1, 1].set_xlabel('Time (µs)')
# axs[1, 1].set_ylabel('Detuning (µm)')
divider4 = make_axes_locatable(axs[1, 1])
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im4, cax=cax4)

# Adjust layout and display the subplots
# plt.tight_layout()
plt.show()

# %% fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

fft_results = []

for data in x:
    X = fft(data)
    fft_results.append(X)

N = len(z[0])
n = np.arange(N)
sr = 1 / np.mean(np.diff(t))
T = N / sr
freq = n / T

n_oneside = N // 2
# get the one side frequency
f_oneside = freq[:n_oneside]

f_fft, d_mesh = np.meshgrid(f_oneside, f)

fft_results = np.abs(fft_results)

fft_results = fft_results[:, :n_oneside]

# %% find f_max
f_max = []
for data in fft_results:
    argmax = np.argmax(data[1:])
    f_max.append(f_oneside[argmax + 1])

# %%  fit for 1d
index = 54
fig, ax = plt.subplots()

x_s = x[index]
y_s = y[index]
r_s = r[index]

tau_guess = 6 * 1e-6
f_guess = 1e6
guess = [1, 1 / f_guess, 0, tau_guess, 0.5]
params = curve_fit(cos_wave_exp, t, r_s, p0=guess, maxfev=200000)[0]

ax.plot(t * 1e6, r_s, label='data')
ax.plot(t * 1e6, cos_wave_exp(t, *params),
        label=f'fit \nDecay = {params[3] * 1e6:.3f} us \nfrequency = {1 / params[1] * 1e-6:.3f} MHz')
ax.set_xlabel('Time [s]')
ax.set_ylabel('State [a.u]')
ax.set_title(f'Spin Locking \n Detuning = {f[index] * 1e-6} MHz')
ax.legend()
plt.show()

# %% fit for all
fig, ax = plt.subplots()

decay_vec_x = []
decay_vec_y = []
decay_vec_r = []
freq_vec_x = []
freq_vec_y = []
freq_vec_r = []
for i in range(59):
    x_s = x[i]
    y_s = y[i]
    r_s = r[i]
    tau_guess = 6
    guess = [1, 1 / np.array(f_max)[i], 0, tau_guess * 1e-6, 0.5]

    params_x = curve_fit(cos_wave_exp, t, x_s, p0=guess, maxfev=200000)[0]
    params_y = curve_fit(cos_wave_exp, t, y_s, p0=guess, maxfev=200000)[0]
    params_r = curve_fit(exp_decay, t, r_s, p0=[1, 5e-6, 0], maxfev=200000)[0]

    freq_x = 1 / params_x[1]
    T2_x = params_x[3]
    freq_y = 1 / params_y[1]
    T2_y = params_y[3]
    T2_r = params_r[1]

    decay_vec_x.append(T2_x)
    freq_vec_x.append(freq_x)

    decay_vec_y.append(T2_y)
    freq_vec_y.append(freq_y)

    decay_vec_r.append(T2_r)

decay_vec_x = np.array(decay_vec_x)
freq_vec_x = np.array(freq_vec_x)
decay_vec_y = np.array(decay_vec_y)
freq_vec_y = np.array(freq_vec_y)
decay_vec_r = np.array(decay_vec_r)

ax.set_xlabel('Detuning [MHz]')
ax.set_ylabel('Decay (µm)')

ax.plot(f * 1e-6, decay_vec_x * 1e6, label='X decay')
ax.plot(f * 1e-6, decay_vec_y * 1e6, label='Y decay')
ax.plot(f * 1e-6, decay_vec_r * 1e6, label='R decay')
plt.legend()

ax.set_title(f" Decay (T2) vs Detuning {qubit}")
ax.set_ylim([0, 7])
plt.show()

# %%

fig, ax = plt.subplots()

pcolormesh = ax.pcolormesh(f_fft[:, 1:] * 1e-6, d_mesh[:, 1:] * 1e-6, fft_results[:, 1:])
ax.set_xlabel('Freq [MHz]')
ax.set_ylabel('Detunings [MHz]')
ax.set_title(f" FFT vs Detuning {qubit}")

ax.plot(freq_vec_x * 1e-6, f * 1e-6, 'x', color='y')

fig.colorbar(pcolormesh, ax=ax, label='Z')
plt.show()
