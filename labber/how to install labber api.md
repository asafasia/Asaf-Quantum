# How to install the Labber API:
written by Guy 12/23

Assuming you have labber installed, you install the API by:

1. find the directory where labber is installed. in my pc in is in C:\Program Files\Keysight\Labber
2. copy the entire Labber directory into the site-packages directory of your interpreter. In my case I paste it in C:\Users\owner\anaconda3\envs\for_lab\Lib\site-packages because I use a conda virtual environment called for_lab. The python interpreter has to be from 3.6 to 3.9  
3. install PyQt5 (with pip install) in the same environment
4. other packages that you might need to install (for me I think they were already installed):
   NumPy,
   h5py,
   qtpy,
   msgpack,
   sip,
   future.
5. to test if the installation worked, run `import Labber` and then `print(Labber.version)`.


In the Labber folder that you coppied earlied there is also a pdf with their user manual. Appendix A in this manual is the documentation of the API.


