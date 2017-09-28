# Classifier for White Dwarf stars by spectral type using their spectra

This is a fun side project I have. during summer 2016, I had an internship in astrophysics at the Université de Montréal under the supervision of prof. Pierre Bergeron, and one of my tasks was to classify White Dwarf stars by their spectral type (hundreds of them) and I remember telling myself that I was certain this could be done more quickly by a computer than a human. [Here](http://patricebechard.github.io/notes/internship-report) is a short internship report I made for that summer. That same summer, the group in which I was working released a large database [the Montreal White Dwarf Database](http://montrealwhitedwarfdatabase.org/) containing spectroscopic and photometric data for more than 30 000 white dwarf stars. I then decided to take a chance and to create that classifying algorithm myself!

The first step of this process is to obtain the data for all the stars. This was fairly easy. The code used to do this named **retrieve_wd_data.py** and uses the list of all the stars in the database, which is named **MWDD-export.csv**. With this method, I obtained 29007 spectras (more than 4gb of data) for different white dwarf stars (for that reason, I won’t upload the data to GitHub).

The next step consists of preparing the data to use is as an input in a neural network.

The third step consists of building and training the neural network that will be used to classify the data.

#### THIS IS A WORK IN PROGRESS!!!