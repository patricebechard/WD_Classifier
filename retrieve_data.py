#!/usr/bin/env python
# -*- coding: ISO-8859-1 -*-
# File: wd_classifier.py
# Author: Patrice Bechard <patricebechard@gmail.com>
# Date: 21.09.2017
# Last Modified: 21.09.2017

# ----------------------------Importing Modules--------------------------------
import numpy as np
import matplotlib.pyplot as plt
from html.parser import HTMLParser
from urllib.request import urlopen, urlretrieve
from urllib import parse

#-----------------------------Functions----------------------------------------

def read_database(database):
    """We put each star of the database and their label in an array"""
    db = open(database)
    stars = []
    nstar = 0
    for line in db:
        stars.append(line.strip().split(','))   #first element in star name, second is spectral type
        nstar += 1
    return stars, nstar

def retrieve_spectra(stars):
    """We visit each webpage for the stars, searching for spectra """
    istar = 0 ; ispectr = 0
    for star in stars:
        url = 'http://montrealwhitedwarfdatabase.org/WDs/' + star[0] + \
                  '/spect.' + star[0] + '.sdss.txt'
        istar += 1
        try:
            response = urlopen(url)
            info = response.readline().decode('iso-8859-1').strip().split(',')
            min_wavelength = float(info[3][15:])
            max_wavelength = float(info[4][15:])
            if min_wavelength < 3900. and max_wavelength > 6700:
                #we keep this spectrum
                urlretrieve(url,'data/spect_%s.txt'%star[0])
            print('SUCCESS,  star %d of %d'%(istar,nstar))
        except:
            #cannot access spectra of wd star (doesn't exist)
            print('*FAILED*, star %d of %d'%(istar,nstar))

#-----------------------------Main---------------------------------------------

database = 'MWDD-export.csv'

stars, nstar = read_database(database)
#retrieve_spectra(stars)


