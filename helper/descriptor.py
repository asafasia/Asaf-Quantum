descriptor = """\
instruments:
  HDAWG:
  - address: DEV8880
    uid: device_hdawg
    interface: usb
  SHFQC:
  - address: DEV12327
    uid: device_shfqc
    interface: usb
  PQSC:
  - address: DEV10164
    uid: device_pqsc
    interface: usb
connections:
  device_hdawg:
    - rf_signal: c13/flux_line
      ports: [SIGOUTS/0]
    - rf_signal: c23/flux_line
      ports: [SIGOUTS/1]
    - rf_signal: c43/flux_line
      ports: [SIGOUTS/2]
    - rf_signal: c53/flux_line
      ports: [SIGOUTS/3]
      

  device_shfqc:
    - iq_signal: q1/drive_line
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q1/drive_line_ef
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q1/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q1/acquire_line
      ports: [QACHANNELS/0/INPUT]

    - iq_signal: q2/drive_line
      ports: SGCHANNELS/1/OUTPUT
    - iq_signal: q2/drive_line_ef
      ports: SGCHANNELS/1/OUTPUT
    - iq_signal: q2/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q2/acquire_line
      ports: [QACHANNELS/0/INPUT]      

    - iq_signal: q3/drive_line
      ports: SGCHANNELS/2/OUTPUT
    - iq_signal: q3/drive_line_ef
      ports: SGCHANNELS/2/OUTPUT
    - iq_signal: q3/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q3/acquire_line
      ports: [QACHANNELS/0/INPUT]

    - iq_signal: q4/drive_line
      ports: SGCHANNELS/3/OUTPUT
    - iq_signal: q4/drive_line_ef
      ports: SGCHANNELS/3/OUTPUT
    - iq_signal: q4/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q4/acquire_line
      ports: [QACHANNELS/0/INPUT]

    - iq_signal: q5/drive_line
      ports: SGCHANNELS/4/OUTPUT
    - iq_signal: q5/drive_line_ef
      ports: SGCHANNELS/4/OUTPUT
    - iq_signal: q5/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q5/acquire_line
      ports: [QACHANNELS/0/INPUT]   
      
         


  device_pqsc:
    - to: device_hdawg
      port: ZSYNCS/0
    - to: device_shfqc
      port: ZSYNCS/1
    - internal_clock_signal


"""
