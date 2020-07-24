#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:57:28 2020

@author: praveenraonittoor
"""

# reference from https://tkf.github.io/2010/10/03/estimate-frequency-using-numpy.html
import pandas as pd
import pylab
import numpy
PI = numpy.pi

class Pisarenko:
    
    def covariance(self,x, k):
        N = len(x) - k
        return (x[:-k] * x[k:]).sum() / N


    def phd1(self,x):
        """
        Estimate frequency using Pisarenko Harmonic Decomposition.
        It returns frequency `omega` in the unit radian/steps.
        If `x[n] = cos(omega*n+phi)` then it returns an estimat of `omega`.
        Note that mean of `x` must be 0.
        See equation (6) from [Kenneth W. K. Lui and H. C. So]_.
        .. [Kenneth W. K. Lui and H. C. So] An Unbiased Pisarenko Harmonic
           Decomposition Estimator For Single-Tone Frequency,
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.75.4859
        """
        r1 = self.covariance(x, 1)
        r2 = self.covariance(x, 2)
        a = (r2 + numpy.sqrt(r2 ** 2 + 8 * r1 ** 2)) / 4 / r1
        ''' if a > 1:  # error should be raised?
            a = 1
        elif a < -1:
            a = -1'''
        return numpy.arccos(a)
    
    
    def freq(self,x, sample_step=1, dt=1.0):
        """Estimate frequency using `phd1`"""
        omega = self.phd1(x[::sample_step])
        return omega / 2.0 / PI / sample_step / dt
    
    
    def plot_x_and_psd_with_estimated_omega(self,x, sample_step=1, dt=1.0):
        y = x[::sample_step]
        F = self.freq(x, sample_step, dt)
    
        pylab.clf()
    
        # plot PSD
        pylab.subplot(211)
        pylab.psd(y, Fs=1.0 / sample_step / dt)
        ylim = pylab.ylim()
        pylab.vlines(F, *ylim)
        pylab.ylim(ylim)
    
        # plot time series
        pylab.subplot(223)
        pylab.plot(x)
    

  

    
dataset_person_1 = pd.read_csv("CGMSeriesLunchPat1.csv").interpolate(axis = 1)
Pisarenko().plot_x_and_psd_with_estimated_omega(dataset_person_1, 20)