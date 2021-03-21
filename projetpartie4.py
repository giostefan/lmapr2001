# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:46:12 2021

@author: Thomas
"""


import numpy as np
import matplotlib.pyplot as plt
from math import pi

def Nslab(omega):
    epsilonInfinity1 = 19.17402283496171
    epsilonInfinity2 = 22.09805938253293
    epsilonInfinity3 = 20.60710234103419

    f11 = 4.213279900760202 
    f21 = 1.961693284797713
    f31 = 0.007966945581503451
    omegaTO11 = 79231717132503.97
    omegaTO21 = 57563096713280.8
    omegaTO31 = 49033016471727.84
    gamma11 = 151929351721.5274
    gamma21 = 151947673655.9077
    gamma31 = 156461516099.8209

    f22 = 2.318085089104123
    f32 = 1.935048918661783
    omegaTO22 = 70373122741473.44
    omegaTO32 = 62330594312793.54
    gamma22 = 151959123549.7535
    gamma32 = 151954274514.8117

    f23 = 3.803690401550578
    f33 = 6.832435225656633
    omegaTO23 = 73806724238431.17
    omegaTO33 = 77733154461442.81
    gamma23 = 151930325428.5151
    gamma33 = 151936692746.3015

    eps11 = epsilonInfinity1 + (f11 * omegaTO11**2)/(omegaTO11**2 - omega**2 - 1j*omega*gamma11)
    eps22 = epsilonInfinity2 + (f21 * omegaTO21**2)/(omegaTO21**2 - omega**2 - 1j*omega*gamma21)+(f22 * omegaTO22**2)/(omegaTO22**2 - omega**2 - 1j*omega*gamma22)+(f23 * omegaTO23**2)/(omegaTO23**2 - omega**2 - 1j*omega*gamma23)
    eps33 = epsilonInfinity3 + (f31 * omegaTO31**2)/(omegaTO31**2 - omega**2 - 1j*omega*gamma31)+(f32 * omegaTO32**2)/(omegaTO32**2 - omega**2 - 1j*omega*gamma32)+(f33 * omegaTO33**2)/(omegaTO33**2 - omega**2 - 1j*omega*gamma33)

    epsilonIso = (eps11+eps22+eps33)/3

    epsilonIsoReal = np.array([x.real for x in epsilonIso])
    epsilonIsoImg = np.array([x.imag for x in epsilonIso])

    n1 = 1/(2**(1/2))*(epsilonIsoReal + (epsilonIsoReal**2+epsilonIsoImg**2)**(1/2))**(1/2)
    k1 = 1/(2**(1/2))*(-epsilonIsoReal + (epsilonIsoReal**2+epsilonIsoImg**2)**(1/2))**(1/2)
    N1 = n1-1j*k1
    return(N1)

def Nmultislab(N0,N1,N2,N):
    Nn = np.zeros(2*N+1)*(1+1j)
    Nn[0]=N0
    Nn[-1]=N0
    for i in range(1,2*N+1):
        if i%2 ==0:
            Nn[i]=N2
        else:
            Nn[i] = N1
    return(Nn)

def calculphi(Nn,N,phi0):
    phi = np.zeros(2*N+1)*(1+1j)
    phi[0]=phi0
    for i in range(1,2*N+1):
        phi[i] = np.arcsin(Nn[i-1]*np.sin(phi[i-1])/Nn[i])
    return(phi)

def fresnel(phi,Nn,N):
    rabp = np.zeros(2*N)*(1+1j)
    rabs = np.zeros(2*N)*(1+1j)
    tabp = np.zeros(2*N)*(1+1j)
    tabs = np.zeros(2*N)*(1+1j)
    for i in range(0,2*N):
        rabp[i] = (Nn[i+1]*np.cos(phi[i])-Nn[i]*np.cos(phi[i+1]))/(Nn[i+1]*np.cos(phi[i])+Nn[i]*np.cos(phi[i+1]))
        rabs[i] = (Nn[i]*np.cos(phi[i])-Nn[i+1]*np.cos(phi[i+1]))/(Nn[i]*np.cos(phi[i])+Nn[i+1]*np.cos(phi[i+1]))
        tabp[i] = (2*Nn[i]*np.cos(phi[i]))/(Nn[i+1]*np.cos(phi[i])+Nn[i]*np.cos(phi[i+1]))
        tabs[i] = (2*Nn[i]*np.cos(phi[i]))/(Nn[i]*np.cos(phi[i])+Nn[i+1]*np.cos(phi[i+1]))
    return(rabp,rabs,tabp,tabs)

def calculbeta(d1,d2,omega,phi,Nn,N,lam):
    beta = np.zeros(2*N-1)*(1+1j)
    for i in range(0,2*N-1):
        if i%2 ==0:
            beta[i]=((2*pi*d1*Nn[i+1])/(lam))*np.cos(phi[i+1])
        else:
            beta[i]=(2*pi*d2*Nn[i+1])/(lam)*np.cos(phi[i+1])
    return(beta)

def calculS(rabp,rabs,tabp,tabs,beta,N):
    Ss = [[1,0],[0,1]]
    Sp = [[1,0],[0,1]]
    for i in range(0,2*N-1):
        Is = [[1/tabs[i],rabs[i]/tabs[i]],[rabs[i]/tabs[i],1/tabs[i]]]
        Ip = [[1/tabp[i],rabp[i]/tabp[i]],[rabp[i]/tabp[i],1/tabp[i]]]
        L = [[np.exp(1j*beta[i]),0],[0,np.exp(-1j*beta[i])]]
        Ss = np.dot(np.dot(Ss,Is),L)
        Sp = np.dot(np.dot(Sp,Ip),L)
    l = 2*N-1   
    Is = [[1/tabs[l],rabs[l]/tabs[l]],[rabs[l]/tabs[l],1/tabs[l]]]
    Ip = [[1/tabp[l],rabp[l]/tabp[l]],[rabp[l]/tabp[l],1/tabp[l]]]
    Ss = np.dot(Ss,Is)
    Sp = np.dot(Sp,Ip)
    return(Ss,Sp)

def calculR(N0,N2,N,phi0,omega,d1,d2,c):
    R = np.zeros(len(omega))
    N1 = Nslab(omega)
    for i in range(0,len(omega)):  
        Nn = Nmultislab(N0,N1[i],N2,N)
        phi = calculphi(Nn,N,phi0)
        rabp,rabs,tabp,tabs = fresnel(phi,Nn,N)
        lam = c*2*pi/omega[i]
        beta = calculbeta(d1,d2,omega,phi,Nn,N,lam)
        Ss,Sp = calculS(rabp,rabs,tabp,tabs,beta,N)
        S11p = (Sp[0])[0]
        S11s = (Ss[0])[0]
        S21p = (Sp[1])[0]
        S21s = (Ss[1])[0]
        Rp = S21p/S11p
        Rs = S21s/S11s
        R[i] = 0.5*(np.abs(Rp)**2+np.abs(Rs)**2)
    return(R)

def calculR2(N0,N2,N,phi0,omega,d1,d2,c,kappa):
    R = np.zeros(len(omega))
    N1 = np.real(Nslab(omega))-1j*kappa
    for i in range(0,len(omega)):  
        Nn = Nmultislab(N0,N1[i],N2,N)
        phi = calculphi(Nn,N,phi0)
        rabp,rabs,tabp,tabs = fresnel(phi,Nn,N)
        lam = c*2*pi/omega[i]
        beta = calculbeta(d1,d2,omega,phi,Nn,N,lam)
        Ss,Sp = calculS(rabp,rabs,tabp,tabs,beta,N)
        S11p = (Sp[0])[0]
        S11s = (Ss[0])[0]
        S21p = (Sp[1])[0]
        S21s = (Ss[1])[0]
        Rp = S21p/S11p
        Rs = S21s/S11s
        R[i] = 0.5*(np.abs(Rp)**2+np.abs(Rs)**2)
    return(R)

def calculT(N0,N2,N,phi0,omega,d1,d2,c):
    T = np.zeros(len(omega))
    N1 = Nslab(omega)
    for i in range(0,len(omega)):  
        Nn = Nmultislab(N0,N1[i],N2,N)
        phi = calculphi(Nn,N,phi0)
        rabp,rabs,tabp,tabs = fresnel(phi,Nn,N)
        lam = c*2*pi/omega[i]
        beta = calculbeta(d1,d2,omega,phi,Nn,N,lam)
        Ss,Sp = calculS(rabp,rabs,tabp,tabs,beta,N)
        S11p = (Sp[0])[0]
        S11s = (Ss[0])[0]
        Tp = 1/S11p
        Ts = 1/S11s
        T[i] = 0.5*(np.abs(Tp)**2+np.abs(Ts)**2)
    return(T)

def calculT2(N0,N2,N,phi0,omega,d1,d2,c,kappa):
    T = np.zeros(len(omega))
    N1 = np.real(Nslab(omega))-1j*kappa
    for i in range(0,len(omega)):  
        Nn = Nmultislab(N0,N1[i],N2,N)
        phi = calculphi(Nn,N,phi0)
        rabp,rabs,tabp,tabs = fresnel(phi,Nn,N)
        lam = c*2*pi/omega[i]
        beta = calculbeta(d1,d2,omega,phi,Nn,N,lam)
        Ss,Sp = calculS(rabp,rabs,tabp,tabs,beta,N)
        S11p = (Sp[0])[0]
        S11s = (Ss[0])[0]
        Tp = 1/S11p
        Ts = 1/S11s
        T[i] = 0.5*(np.abs(Tp)**2+np.abs(Ts)**2)
    return(T)

def calculA(R,T):
    A = 1-R-T
    return(A)

#calcul de l'absorbance
N0 = 1-1j*0
N2 = 1-1j*0
N = 3
d1 = 90*10**-9
d2 = 400*10**-9
c = 3*10**8
phi0 = np.linspace(0,(pi/2)-0.001,100)
#v = np.linspace(0,0.2*c,100)
#beta1 = v/c
#omega = 0.9*(1.338*10**15)*np.sqrt((1-beta1)/(1+beta1))
omega = np.linspace(10**12,0.9*1.338*10**15,1000)

A = np.zeros((len(phi0),len(omega)))
for i in range(0,len(phi0)):
    R = calculR(N0,N2,N,phi0[i],omega,d1,d2,c)
    T = calculT(N0,N2,N,phi0[i],omega,d1,d2,c)
    A[i,:] = calculA(R,T)

I0 = 10e9
S = 10
Apower = A*I0*S

#calcul de l'émissivité hémisphérique
eps = A #loi de Kirchoff

epshemi = np.zeros(len(omega))
for j in range(0,len(omega)):
    x = eps[:,j]*np.cos(phi0)*np.sin(phi0)
    y = phi0
    epshemi[j] = 2*pi*np.trapz(x,y)
    
#calcul du black body
    
Tbbd = 1000 #[K]
h = 6.626*10**-34 #constante de Planck
k = 1.38*10**-23 #constante de boltzmann
lam = c*2*pi/omega
bbd = (2*h*c**2)/(lam**5*(np.exp((h*c)/(k*Tbbd*lam))-1))


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(omega[0:999], bbd[0:999]*10**-9, color='r')
ax2.plot(omega[0:999], epshemi[0:999], color='b')
ax1.set_xlabel('Frequency [rad/sec]')
ax2.set_xscale('log')
ax1.set_ylim(0,5)
ax2.set_ylim(0,2)
ax2.set_ylabel('emissivity [/]', color='b')
ax1.set_ylabel('bbd [/]', color='r')
plt.show()

#calcul du black body T variable
Tbbd2 = np.linspace(1,1500,1000)
bbd2 = np.zeros((len(omega),len(Tbbd2)))
for index in range(0,len(Tbbd2)):
    bbd2[index,:] = (2*h*c**2)/(lam**5*(np.exp((h*c)/(k*Tbbd2*lam))-1))

#radiated power
rp = np.zeros(len(Tbbd2))
for index2 in range(0,len(Tbbd2)):
    cal = epshemi*bbd2[:,index2]
    rp[index2] = -2*S*np.trapz(cal,lam)

#absorbed power
kappa = 10**-5    #ligne à modifier

Aa = np.zeros(len(Tbbd2))
Ra = calculR2(N0,N2,N,phi0[0],omega,d1,d2,c,kappa)
Ta = calculT2(N0,N2,N,phi0[0],omega,d1,d2,c,kappa)
Aa[:] = calculA(Ra[-1],Ta[-1])

Apower2 = Aa*I0*S
    
plt.plot(Tbbd2,Apower2*10**-5)
plt.plot(Tbbd2,rp*10**-5)
plt.xlabel("Temperature [K]")

intersec = 10
index3 = 0
for i2 in range(0,len(Tbbd2)):
    if intersec > np.abs(Apower2[i2]*10**-5-rp[i2]*10**-5):
        index3 = i2
print(Tbbd2[index3])
    