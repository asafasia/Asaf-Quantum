from laboneq.simple import *
# from laboneq.dsl.experiment.pulse_library import register_pulse_functional
import numpy as np

from qubit_parameters import *


@pulse_library.register_pulse_functional
def lorentizan(
    x, p,n,**_
):
    
    a = np.sqrt((1/p)**(1/n) - 1)
    
    
    lorentizan = 1/(1+(a*x)**2)**n
 
    
    return lorentizan


@pulse_library.register_pulse_functional
def k_pulse(x,**_):
    return np.heaviside(x-0.5,0.5)*0
    # return np.ones_like(x)




def pi_pulse(qubit):
    
    pi_pulse = pulse_library.const(
        uid=f"pi_pulse_{qubit}",
        length=qubit_parameters[qubit]["pi_len"],
        amplitude=qubit_parameters[qubit]["pi_amp"],
        can_compress=True
        )
    
    return pi_pulse


def many_pi_pulse(qubit,pis): 
    many_pi_pulse = pulse_library.const(
        uid=f"pi_pulse_{qubit}",
        length=qubit_parameters[qubit]["pi_len"],
        amplitude=qubit_parameters[qubit]["pi_amp"]*pis
        )
   

    return many_pi_pulse


def power_broadening_pulse(qubit,
                           amplitude = None,
                           length = None,
                           pulse_type = 'Square',
                           p=50,
                           n=2/3,
                           
                           ):
    
    
    if not amplitude:
        amplitude = qubit_parameters[qubit]["pi_amp"]
    
    if not length:
        length = qubit_parameters[qubit]["pi_len"]

    
    
    if pulse_type == 'Square':
    
        pulse = pulse_library.const(
            uid=f"pi_pulse_{qubit}",
            length=length,
            amplitude=amplitude
            )
        
    elif pulse_type == 'Gaussian':
        pulse = pulse_library.gaussian(
            uid=f"pi_pulse_{qubit}",
            sigma = np.sqrt(-np.ln(t)/2),
            length=length,
            amplitude=amplitude
            )
            
    elif pulse_type == 'Lorentzian':
        pulse = lorentizan(
            uid=f"pi_pulse_{qubit}",
            length=length,
            amplitude=amplitude,
            p = p,
            n = n
            )
    else:
        print("Enetred worng pulse")
        
    return pulse



def kernel_pulse(qubit):
    kernel_pulse = pulse_library.const(
        uid=f"kernel_pulse_{qubit}",
        length=qubit_parameters[qubit]["res_len"],
        amplitude=qubit_parameters[qubit]["res_amp"]
        )
    
    return kernel_pulse
    


def spec_pulse(qubit):
    spec_pulse = pulse_library.const(
        uid=f"spec_pulse_{qubit}",
        length=qubit_parameters[qubit]["drive_len"],
        amplitude=qubit_parameters[qubit]["drive_amp"]
        )
    
    return spec_pulse
    
def readout_pulse(qubit):
    readout_pulse = pulse_library.const(
        uid = f"readout_pulse_{qubit}",
        length = qubit_parameters[qubit]["res_len"],
        amplitude = qubit_parameters[qubit]["res_amp"]
        )
    
    # readout_pulse = pulse_library.gaussian(
    #     uid = f"readout_pulse_{qubit}",
    #     length = qubit_parameters[qubit]["res_len"],
    #     amplitude = qubit_parameters[qubit]["res_amp"],
    #     sigma = 1/3,
    #     order = 30
    #     )
    return readout_pulse

def flux_pulse(qubit):
    flux_pulse = pulse_library.const(
        uid=f"flux_pulse_{qubit}",
        length=120e-6,
        amplitude=1,
        can_compress=True

        
        )
    
    # flux_pulse = pulse_library.gaussian(
    #     uid = f"flux_pulse_{qubit}",
    #     length = 120e-6,
    #     amplitude = 1,
    #     sigma = 1/3,
    #     order = 30,
    #     width = width
        

        
    #     )
    
    return flux_pulse
