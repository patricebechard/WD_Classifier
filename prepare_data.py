#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: preprocess_data.py
# Author: Patrice Bechard <patricebechard@gmail.com>
# Date: 01.10.2017
# Last Modified: 01.10.2017

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import lfilter

np.set_printoptions(threshold=sys.maxsize)

#---------------------------Functions-----------------------------------------
def smooth_spectrum(data, figname, n=15, a=1, lower=3900, upper=6800, npts=1000):

    b = [1.0 / n] * n
    yy = lfilter(b,a,data[1])                   #smoothens spectrum
    smooth_x = np.linspace(lower,upper,npts)
    smooth_y = np.interp(smooth_x, data[0],yy)  #always same grid for data

    params = np.polyfit(smooth_x, smooth_y, 2)  #fit a 2nd degree polynomial
    nor = params[2] + smooth_x * params[1] + (smooth_x ** 2) * params[0]
    smooth_y /= nor              #normalized spectrum (not so clean)

    smooth_y -= np.mean(smooth_y)               #center data to 0
    smooth_y /= np.std(smooth_y)                #scale between 1 and -1 mostly

  #  b = [1.0 / n] * n
  #  smooth_y = lfilter(b,a,smooth_y)            #second smoothing

    return smooth_y

#---------------------------Main----------------------------------------------

if __name__ == "__main__":

    database = "MWDD-export.save"
    database = open(database)
    i = 0

    kept_types = ['DA','DB','DO','DC','DQ','DZ','DAB','DAZ','DA+M','DAM','DAH','DBZ']

    for star in database:
        starname, sptype = star.strip().split(',')
        try:
            filename = "data/spect_%s.txt"%starname
            data = np.loadtxt(filename,delimiter=',',skiprows=2)
            prepared_data = smooth_spectrum(data.T, starname)

            label = [i for i in range(len(kept_types)) if sptype == kept_types[i]]
            prepared_data = np.append(prepared_data,label)

            prepared_data = np.expand_dims(prepared_data,axis=0)

            if i == 0:
                prepared_db = prepared_data
            else:
                prepared_db = np.append(prepared_db,prepared_data,axis=0)

            i += 1
            if i % 100 == 0:
                print(i)
            if i == 5000:
                np.savetxt('test.csv',prepared_db,delimiter=',')
                sys.exit()

        except FileNotFoundError:
            """The given file doesn't exist, the spectra is not available"""
            pass

    np.savetxt('prepared_db.csv',prepared_db,delimiter=',')
