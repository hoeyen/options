import numpy as np
from scipy.stats import norm

#s = spot price
#k = strike
#rate = risk-free rate
#vol = volatility
#div = dividend rate

def dOne(s, k, t, rate, vol, div):
    dOne = (np.log(s/k) + (rate-div+0.5* vol**2)*t)/(vol * (t**(0.5)))
    return dOne

def NdOne(s, k, t, rate, vol, div):
    NdOne = np.exp(-(dOne(s, k, t, rate, vol, div)**2)/2) / ((2 * np.pi)**(0.5))
    return NdOne

def dTwo(s, k, t, rate, vol, div):
    dTwo = dOne(s, k, t, rate, vol, div) - vol * (t**(0.5))
    return dTwo

def NdTwo(s, k, t, rate, vol, div):
    NdTwo = norm.cdf(dTwo(s, k, t, rate, vol, div))
    return NdTwo

def CallOption(s, k, t, rate, vol, div):
    CallOption = np.exp(-div*t)*s*norm.cdf(dOne(s,k,t,rate,vol,div))-k*np.exp(-rate*t)*norm.cdf(dOne(s,k,t,rate,vol,div)-vol*(t**(0.5)))
    return CallOption

def PutOption(s, k, t, rate, vol, div):
    PutOption = k * np.exp(-rate * t) * norm.cdf(-dTwo(s, k, t, rate, vol, div)) - np.exp(-div * t) * s * norm.cdf(-dOne(s, k, t, rate, vol, div))
    return PutOption

def CallDelta(s, k, t, rate, vol, div):
    CallDelta = norm.cdf(dOne(s, k, t, rate, vol, div))
    return CallDelta

def PutDelta(s, k, t, rate, vol, div):
    PutDelta = norm.cdf(dOne(s, k, t, rate, vol, div)) - 1
    return PutDelta

def CallTheta(s, k, t, rate, vol, div):
    CT = -(s * vol * NdOne(s, k, t, rate, vol, div)) / (2 * (t**(0.5))) - rate * k * np.exp(-rate*t) * NdTwo(s, k, t, rate, vol, div)
    CallTheta = CT / 365
    return CallTheta

def Gamma(s, k, t, rate, vol, div):
    Gamma = NdOne(s, k, t, rate, vol, div) / (s * (vol * (t**(0.5))))
    return Gamma

def Vega(s, k, t, rate, vol, div):
    Vega = 0.01 * s * (t**(0.5)) * NdOne(s, k, t, rate, vol, div)
    return Vega

def PutTheta(s, k, t, rate, vol, div):
    PT = -(s * vol * NdOne(s, k, t, rate, vol, div)) / (2 * (t**(0.5))) + rate * k * np.exp(-rate * (t)) * (1 - NdTwo(s, k, t, rate, vol, div))
    PutTheta = PT / 365
    return PutTheta

def CallRho(s, k, t, rate, vol, div):
    CallRho = 0.01 * k * t * np.exp(-rate * t) * norm.cdf(dTwo(s, k, t, rate, vol, div))
    return CallRho

def PutRho(s, k, t, rate, vol, div):
    PutRho = -0.01 * k * t * np.exp(-rate * t) * (1 - norm.cdf(dTwo(s, k, t, rate, vol, div)))
    return PutRho