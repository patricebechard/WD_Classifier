#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: preprocess_data.py
# Author: Patrice Bechard <patricebechard@gmail.com>
# Date: 01.10.2017
# Last Modified: 01.10.2017

#---------------------------Functions-----------------------------------------

#---------------------------Main----------------------------------------------

if __name__ == "__main__":

    database = "MWDD-export.csv"
    database = open(database)

<<<<<<< HEAD
    newdatabase = open('MWDD-export.save','w')
=======
    newdatabase = open('MWDD-export2.csv','w')
>>>>>>> 063f4ace563e8d61c0c034d58193ef9fa6054f59

    for line in database:
        star, sptype = line.strip().split(',')
        chars_to_remove = ':._?()0123456789abcdefghijklmnopqrstuvwxyz\%'
        for char in chars_to_remove:
            sptype = sptype.replace(char,'')
        sptype = sptype.replace('-','+')

        kept_types = ['DA','DB','DO','DC','DQ','DZ','DAB','DAZ','DA+M','DAM','DAH','DBZ']

        if sptype in kept_types:
            newdatabase.write(star+','+sptype+'\n')


