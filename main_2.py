import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Function to model the resonance curve (Lorentzian)
def lorentzian(f, A, f0, delta_f):
    return A / (1 + ((f - f0) / (delta_f / 2)) ** 2)


# Parameters for the resonance curve
A = 1  # Amplitude
f0 = 100  # Resonant frequency (in Hz)
delta_f = 10  # Bandwidth (in Hz)

# Generating the frequency range
frequencies = np.linspace(f0 - 50, f0 + 50, 100)

# Generating the resonance curve with noise
noise = np.random.normal(0, 1 / 6, frequencies.shape)
resonance_curve = lorentzian(frequencies, A, f0, delta_f) + noise

print(f'var = {np.var(resonance_curve)}')
print(f'max = {np.max(resonance_curve)}')

print(np.max(resonance_curve) / np.var(resonance_curve))

# Fit the data to the Lorentzian to find f0 and delta_f accurately
popt, pcov = curve_fit(lorentzian, frequencies, resonance_curve, p0=[A, f0, delta_f])

# Extracted parameters
fitted_A, fitted_f0, fitted_delta_f = popt

# Calculate the Q factor
Q_factor = fitted_f0 / fitted_delta_f

f_max = frequencies[np.argmax(resonance_curve)]
# Plotting the data and the fit
plt.figure(figsize=(10, 6))
plt.plot(frequencies, resonance_curve, label='Noisy Resonance Data')
plt.plot(frequencies, lorentzian(frequencies, *popt), label='Fitted Curve', linestyle='--')
plt.axvline(x=f_max, color='red')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Resonance Curve with Q Factor = {Q_factor:.2f}')
plt.legend()
plt.grid()
plt.show()
