
def play_flux_qdac(qubit,flux_bias):
    
    
    qubit_dict = {
        'q1':1,
        'q2':2,
        'q3':3,
        'q4':4,
        'q5':5,
                  }
    
    
    
    import qdac as qdac

    QDAC_port='COM4'
    def set_and_show_QDAC(qdac, channel, voltage): #added by Naftali
        print("Setting voltage to %f" % voltage)
        qdac.setDCVoltage(channel, voltage)
        print(f"Measured voltage is {qdac.getDCVoltage(channel)}")
        print(f"Measured current is {qdac.getCurrentReading(channel)}")
          
    #For channel 5:
    with qdac.qdac(QDAC_port) as qdac:
          qdac.setVoltageRange(5, 10) # [V]
          qdac.setCurrentRange(5, 1e-4) #[A]
          set_and_show_QDAC(qdac, qubit_dict[qubit], flux_bias)

if __name__ == "__main__":
    play_flux_qdac('q2', 0.0001)