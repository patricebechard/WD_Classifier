White Dwart Star classifier using Machine Learning

This project consists of building an algorithm capable of classifying white dwarf stars based on their spectrum. The data used is extracted from the Montreal White Dwarf Database (MWDD)

Range of wavelength of spectrum used : [3900, 7000] Angstrom

Numbers of WD in database as of 2017/09/21 : 31180

program consists of :

accessing MWDD for spectra of stars in list of stars
using those spectra as the input for the neural network
using the spectral type of the stars as labels


regularize the data before fitting (mean is 0, span from -1 to 1)
