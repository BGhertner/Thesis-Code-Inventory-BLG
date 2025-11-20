#
# Ben Ghertner 2025
#
# Compute the nonlinear Schrodinger equation parameter (gamma)
# over a range of wavenumber (k) and latent heat response (L) values
#      
#       - Figure 4.6
#
# Note: change the parameter loaddata in the plotting section if
#       you want to generate new data for a different parameter space.
#       The list of zero's of M0 will need to be updated as well if
#       you change the range of k values used.

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = True
plt.rcParams["font.size"] = 11
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}"

# Uses secant method to numerically solve the dispersion relationship Det(M) = 0 
# with M given in Eq (3.44) for the frequency of the wave
#
# input:
#       om_0 - first guess at frequency
#       N2c  - buoyancy frequency in cloud
#       N2i  - buoyancy frequency in inversion (clear)
#       N2t  - buoyancy frequency in troposphere (above inversion)
#       Hc   - nominal height of cloud edge 
#       Hi   - nominal height of inversion edge
#       HT   - domain top height 
# returns:
#       om   - frequency (numerically) satisfying Det(M) = 0
def secant_omega(om_0, N2c, N2i, N2t, Hc, Hi, HT, k):
    # Dispersion relationship function Det(M) 
    # as a function of the frequency
    #
    # input:
    #       om - frequency
    # returns:
    #       Det(M) from Eq (3.44)
    def f(om):
        #Vertical wavenumbers
        mt = k*np.sqrt(1 - N2t/om**2)
        mi = k*np.sqrt(N2i/om**2 - 1)
        mc = k*np.sqrt(1 - N2c/om**2)
        
        #as in Eq (3.44)
        M = np.array([[mt*np.sin(mi*Hi)/np.tanh(mt*(Hi-HT)) - mi*np.cos(mi*Hi),
                       mt*np.cos(mi*Hi)/np.tanh(mt*(Hi-HT)) + mi*np.sin(mi*Hi)],

                      [mi*np.cos(mi*Hc) - mc*np.sin(mi*Hc)/np.tanh(mc*Hc),
                      -mi*np.sin(mi*Hc) - mc*np.cos(mi*Hc)/np.tanh(mc*Hc)]])
        return np.linalg.det(M)
    #Initialize first guesses at the frequency
    om_m1 = om_0
    om_m2 = om_0*(1.001)
    #Iterate until tolerance is met
    while(np.abs((om_m1-om_m2)/om_m1)>1e-15):
        #Save last guess
        temp = om_m1
        #Secant step
        om_m1 = om_m1 - f(om_m1)*(om_m1-om_m2)/(f(om_m1)-f(om_m2))
        #update last guess
        om_m2 = temp

    return om_m1


#sympy symbols (variables)
k, z, x, t, X, T, L = sym.symbols('k, z, x, t, X, T, L', real=True)
om, om_p = sym.symbols("omega omega_k", real=True)
c = om/k
I = sym.I #sympy imaginary unit for convience

# From Table 4.1
N2c = (2*np.pi)**2
N2i = 2*(2*np.pi)**2
N2t = np.pi**2

Hc = 2.
Hi = 4.
HT = 10.0

mt = k*sym.sqrt(1 - N2t/om**2)
mi = k*sym.sqrt(N2i/om**2 - 1)
mc = k*sym.sqrt(1 - N2c/om**2)

#Default values of k and omega
k_val  = 2.0
#Find frequency for the given wavenumber and parameters
om_val = secant_omega(2.5*np.pi+0.j, N2c=N2c, N2i=N2i, N2t=N2t, Hc=Hc, Hi=Hi, HT=HT, k=k_val)
print(f'omega: {om_val:.5f}')
print(f'mt:    {mt.subs({k:k_val,om:om_val}):.5f}')
print(f'mi:    {mi.subs({k:k_val,om:om_val}):.5f}')
print(f'mc:    {mc.subs({k:k_val,om:om_val})}')


# Helpful function to plug in k and omega values and evaluate
#
# inputs:
#       f      - sympy expression to be evaluated
#       z      - evalutation points
#       k_val  - numerical value of the horizontal wavenumber
#       om_val - numerical value of the frequency of the wave
#       L_val  - latent heat response
# returns:
#       f_eval - the expression f evaluated at the given points
def sym_to_eval(f, zz, k_val, om_val, L_val=0.5):
    val_dict = {
        k:   k_val,
        om:  om_val,
        L :  L_val
    }
    return sym.lambdify(z, f.subs(val_dict))(zz)


# Given a value of k and omega compute the nonlinear NLS parameter gamma Eq (4.58).
# This routine is a slimmed down version of the computation in nl_wave_sol.ipynb.
# See nl_wave_sol.ipynb for more detailed comments in code.
#
# inputs:
#       k_val  - numerical value of the horizontal wavenumber
#       om_val - numerical value of the frequency of the wave
# returns:
#       gamma  - NLS nonlinear coefficient
def get_gamma(k_val, om_val):
    a = mt*sym.sin(mi*Hi)/sym.tanh(mt*(Hi-HT)) - mi*sym.cos(mi*Hi)
    b = mt*sym.cos(mi*Hi)/sym.tanh(mt*(Hi-HT)) + mi*sym.sin(mi*Hi)

    c = mi*sym.cos(mi*Hc) - mc*sym.sin(mi*Hc)/sym.tanh(mc*Hc)
    d =-mi*sym.sin(mi*Hc) - mc*sym.cos(mi*Hc)/sym.tanh(mc*Hc)

    lambda2 = (1/2)*(sym.sqrt((a - d)**2 + 4*b*c) + a + d).subs({k:   k_val, om:  om_val})
    v2 = [
        (a - d + sym.sqrt((a - d)**2 + 4*b*c))/(2*c),
        1
    ]

    lambda1 = (1/2)*(-sym.sqrt((a - d)**2 + 4*b*c) + a + d).subs({k:   k_val, om:  om_val})
    v1 = [
        (a - d - sym.sqrt((a - d)**2 + 4*b*c))/(2*c),
        1
    ]

    if np.abs(lambda1) > np.abs(lambda2):
        
        temp = [v2[0], v2[1]]
        v2 = [v1[0], v1[1]]
        v1 = temp

    #Leading Order Solution
    psia_t = sym.sin(mi*Hi)*sym.sinh(mt*(z-HT))/sym.sinh(mt*(Hi-HT))
    psia_i = sym.sin(mi*z)
    psia_c = sym.sin(mi*Hc)*sym.sinh(mc*z)/sym.sinh(mc*Hc)

    psib_t = sym.cos(mi*Hi)*sym.sinh(mt*(z-HT))/sym.sinh(mt*(Hi-HT))
    psib_i = sym.cos(mi*z)
    psib_c = sym.cos(mi*Hc)*sym.sinh(mc*z)/sym.sinh(mc*Hc)

    psih_0_t = v1[0]*psia_t + v1[1]*psib_t
    psih_0_i = v1[0]*psia_i + v1[1]*psib_i
    psih_0_c = v1[0]*psia_c + v1[1]*psib_c

    #Build phi function
    phi_t = v2[0]*psia_t + v2[1]*psib_t
    phi_i = v2[0]*psia_i + v2[1]*psib_i
    phi_c = v2[0]*psia_c + v2[1]*psib_c

    #phi derivative jump at inversion
    J_dphi = sym_to_eval(sym.diff(phi_t,z), Hi, k_val, om_val)-sym_to_eval(sym.diff(phi_i,z), Hi, k_val, om_val)

    #print(f'Derivative jump in phi at inversion: {J_dphi}')



    #Compute psi^X, psi^T
    psih_X_t = -I*sym.diff(psih_0_t, k)
    psih_X_i = -I*sym.diff(psih_0_i, k)
    psih_X_c = -I*sym.diff(psih_0_c, k)

    psih_T_t =  I*sym.diff(psih_0_t, om)
    psih_T_i =  I*sym.diff(psih_0_i, om)
    psih_T_c =  I*sym.diff(psih_0_c, om)

    # Add phi to make derivative continuity at Hi
    del_X_Hi = sym_to_eval(sym.diff(psih_X_t,z), Hi, k_val, om_val)-sym_to_eval(sym.diff(psih_X_i,z), Hi, k_val, om_val)
    del_T_Hi = sym_to_eval(sym.diff(psih_T_t,z), Hi, k_val, om_val)-sym_to_eval(sym.diff(psih_T_i,z), Hi, k_val, om_val)

    alpha_X = -del_X_Hi/J_dphi
    alpha_T = -del_T_Hi/J_dphi

    psih_X_t += alpha_X*phi_t
    psih_X_i += alpha_X*phi_i
    psih_X_c += alpha_X*phi_c

    psih_T_t += alpha_T*phi_t
    psih_T_i += alpha_T*phi_i
    psih_T_c += alpha_T*phi_c

    del_X = sym_to_eval(sym.diff(psih_X_i,z), Hc, k_val, om_val)-sym_to_eval(sym.diff(psih_X_c,z), Hc, k_val, om_val)
    del_T = sym_to_eval(sym.diff(psih_T_i,z), Hc, k_val, om_val)-sym_to_eval(sym.diff(psih_T_c,z), Hc, k_val, om_val)

    om_p = del_X/del_T

    #print(f'group velocity = {om_p}')








    #Nonlinear terms (1,2)
    m2t = k*sym.sqrt(4 - N2t/om**2)
    m2i = k*sym.sqrt(N2i/om**2 - 4)
    m2c = k*sym.sqrt(4 - N2c/om**2)

    psi2a_t = sym.sin(m2i*Hi)*sym.sinh(m2t*(z-HT))/sym.sinh(m2t*(Hi-HT))/sym.sin(m2i*Hi)
    psi2a_i = sym.sin(m2i*z)/sym.sin(m2i*Hi)
    psi2a_c = sym.sin(m2i*Hc)*sym.sinh(m2c*z)/sym.sinh(m2c*Hc)/sym.sin(m2i*Hi)

    psi2b_t = sym.sinh(m2t*(z-HT))/sym.sinh(m2t*(Hi-HT))/sym.cos(m2i*(Hc-Hi))
    psi2b_i = sym.cos(m2i*(z-Hi))/sym.cos(m2i*(Hc-Hi))
    psi2b_c = sym.sinh(m2c*z)/sym.sinh(m2c*Hc)

    J_dpsi2a_Hi = sym.diff(psi2a_t, z).subs({z:Hi}) - sym.diff(psi2a_i, z).subs({z:Hi})
    J_dpsi2b_Hi = sym.diff(psi2b_t, z).subs({z:Hi}) - sym.diff(psi2b_i, z).subs({z:Hi})
    J_dpsi2a_Hc = sym.diff(psi2a_i, z).subs({z:Hc}) - sym.diff(psi2a_c, z).subs({z:Hc})
    J_dpsi2b_Hc = sym.diff(psi2b_i, z).subs({z:Hc}) - sym.diff(psi2b_c, z).subs({z:Hc})

    M12 = sym.Matrix([[J_dpsi2a_Hi, J_dpsi2b_Hi],
                    [J_dpsi2a_Hc, J_dpsi2b_Hc]])

    J_dpsi12_Hi = -k**3/om**3*(N2t-N2i)*(psih_0_i.subs({z:Hi}))**2
    J_dpsi12_Hc = -(1+L)*k**3/om**3*(N2i-N2c)*(psih_0_i.subs({z:Hc}))**2

    W12 = M12.inv()@sym.Matrix([[J_dpsi12_Hi], [J_dpsi12_Hc]])

    psih_12_i = W12[0,0]*psi2a_i + W12[1,0]*psi2b_i


    #Nonlinear terms (1,0)
    Nt = np.sqrt(N2t)
    Ni = np.sqrt(N2i)
    Nc = np.sqrt(N2c)

    m0t = Nt/om_p
    m0i = Ni/om_p
    m0c = Nc/om_p

    #First construct a continuous particular solution to the ODE
    psi10_pa_t = mt*sym.sin(mi*Hi)**2*sym.sinh(2*mt*(z-HT))/sym.sinh(mt*(Hi-HT))**2/(m0t**2 + 4*mt**2)
    psi10_pa_i = mi*sym.sin(2*mi*z)/(m0i**2 - 4*mi**2)
    psi10_pa_c = mc*sym.sin(mi*Hc)**2*sym.sinh(2*mc*z)/sym.sinh(mc*Hc)**2/(m0c**2 + 4*mc**2)

    psi10_pb_t = mt*sym.cos(mi*Hi)**2*sym.sinh(2*mt*(z-HT))/sym.sinh(mt*(Hi-HT))**2/(m0t**2 + 4*mt**2)
    psi10_pb_i = -mi*sym.sin(2*mi*z)/(m0i**2 - 4*mi**2)
    psi10_pb_c = mc*sym.cos(mi*Hc)**2*sym.sinh(2*mc*z)/sym.sinh(mc*Hc)**2/(m0c**2 + 4*mc**2)

    psi10_pab_t = mt/2*sym.sin(2*mi*Hi)*sym.sinh(2*mt*(z-HT))/sym.sinh(mt*(Hi-HT))**2/(m0t**2 + 4*mt**2)
    psi10_pab_i = mi*sym.cos(2*mi*z)/(m0i**2 - 4*mi**2)
    psi10_pab_c = mc/2*sym.sin(2*mi*Hc)*sym.sinh(2*mc*z)/sym.sinh(mc*Hc)**2/(m0c**2 + 4*mc**2)

    psi10_pd_t = m0t**2*k/om*(1+2*k/om*om_p)*(1-k/om*om_p)*(v1[0]**2*psi10_pa_t + 2*v1[0]*v1[1]*psi10_pab_t + v1[1]**2*psi10_pb_t)
    psi10_pd_i = m0i**2*k/om*(1+2*k/om*om_p)*(1-k/om*om_p)*(v1[0]**2*psi10_pa_i + 2*v1[0]*v1[1]*psi10_pab_i + v1[1]**2*psi10_pb_i)
    psi10_pd_c = m0c**2*k/om*(1+2*k/om*om_p)*(1-k/om*om_p)*(v1[0]**2*psi10_pa_c + 2*v1[0]*v1[1]*psi10_pab_c + v1[1]**2*psi10_pb_c)

    J_psi10_pd_Hi = psi10_pd_t.subs({z:Hi}) - psi10_pd_i.subs({z:Hi})
    J_psi10_pd_Hc = psi10_pd_i.subs({z:Hc}) - psi10_pd_c.subs({z:Hc})

    psi_10_p_t = psi10_pd_t + (-J_psi10_pd_Hi + sym.sin(m0i*Hi))*sym.sin(m0t*(z-HT))/sym.sin(m0t*(Hi-HT))
    psi_10_p_i = psi10_pd_i + sym.sin(m0i*z)
    psi_10_p_c = psi10_pd_c + (J_psi10_pd_Hc + sym.sin(m0i*Hc))*sym.sin(m0c*z)/sym.sin(m0c*Hc)

    J_dpsi10_p_Hi = sym.diff(psi_10_p_t, z).subs({z:Hi}) - sym.diff(psi_10_p_i, z).subs({z:Hi})
    J_dpsi10_p_Hc = sym.diff(psi_10_p_i, z).subs({z:Hc}) - sym.diff(psi_10_p_c, z).subs({z:Hc})

    #Continuous homogeneous solutions
    psi10a_t = sym.sin(m0i*Hi)*sym.sin(m0t*(z-HT))/sym.sin(m0t*(Hi-HT))
    psi10a_i = sym.sin(m0i*z)
    psi10a_c = sym.sin(m0i*Hc)*sym.sin(m0c*z)/sym.sin(m0c*Hc)

    psi10b_t = sym.cos(m0i*Hi)*sym.sin(m0t*(z-HT))/sym.sin(m0t*(Hi-HT))
    psi10b_i = sym.cos(m0i*z)
    psi10b_c = sym.cos(m0i*Hc)*sym.sin(m0c*z)/sym.sin(m0c*Hc)

    #Build the solution with the correct derivative jump conditions
    J_dpsi10a_Hi = sym.diff(psi10a_t, z).subs({z:Hi}) - sym.diff(psi10a_i, z).subs({z:Hi})
    J_dpsi10b_Hi = sym.diff(psi10b_t, z).subs({z:Hi}) - sym.diff(psi10b_i, z).subs({z:Hi})
    J_dpsi10a_Hc = sym.diff(psi10a_i, z).subs({z:Hc}) - sym.diff(psi10a_c, z).subs({z:Hc})
    J_dpsi10b_Hc = sym.diff(psi10b_i, z).subs({z:Hc}) - sym.diff(psi10b_c, z).subs({z:Hc})

    M10 = sym.Matrix([[J_dpsi10a_Hi, J_dpsi10b_Hi],
                    [J_dpsi10a_Hc, J_dpsi10b_Hc]])

    J_dpsi10_Hi = -2*k**3/om**3*(N2t-N2i)*(psih_0_i.subs({z:Hi}))**2
    J_dpsi10_Hc = -2*k**3/om**3*(N2i-N2c)*(1+L)*(psih_0_i.subs({z:Hc}))**2

    W10 = M10.inv()@(sym.Matrix([[J_dpsi10_Hi], [J_dpsi10_Hc]]) - sym.Matrix([[J_dpsi10_p_Hi], [J_dpsi10_p_Hc]]))

    psih_10_i = psi_10_p_i + W10[0,0]*psi10a_i + W10[1,0]*psi10b_i

    #Other parts of the solution
    Zih_10     = -1/om_p*psih_10_i.subs({z:Hi}) + k/om/om_p*sym.diff(psih_0_i**2, z).subs({z:Hi})
    Zch_10     = -(1+L)/om_p*psih_10_i.subs({z:Hc}) + (1+L)*k/om*(1/om_p + 1/2*k/om*L)*sym.diff(psih_0_i**2, z).subs({z:Hc}) \
                +I/om_p*k**2/om*(1+L)*L*(psih_0_i*sym.diff(psih_X_i, z) - psih_X_i*sym.diff(psih_0_i, z)).subs({z:Hc}) \
                -I*k**2/om*(1+L)*L*(psih_0_i*sym.diff(psih_T_i, z) - psih_T_i*sym.diff(psih_0_i, z)).subs({z:Hc})




    #Nonlinear terms (2,1)
    #Correct jump in this mode at each Hi and Hc
    J_psi21_Hi = 3/2*k**4/om**4*(N2t - N2i)*(psih_0_i.subs({z:Hi}))**3 + k/om*(psih_0_i.subs({z:Hi}))*(J_dpsi10_Hi + J_dpsi12_Hi)
    J_psi21_Hc = 3/2*(1+L)**2*k**4/om**4*(N2i - N2c)*(psih_0_i.subs({z:Hc}))**3 + (1+L)*k/om*(psih_0_i.subs({z:Hc}))*(J_dpsi10_Hc + J_dpsi12_Hc)

    #Particular solutions to each forcing

    #6 for the 0-0 interaction
    psi21_p0aa_t = mt**2*sym.sin(mi*Hi)**3/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p0aa_i = mi**2*(-sym.sin(3*mi*z)/(8*mi**2) + z*sym.cos(mi*z)/(2*mi))
    psi21_p0aa_c = mc**2*sym.sin(mi*Hc)**3/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p0ba_t = mt**2*sym.sin(mi*Hi)**2*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p0ba_i = mi**2*(-sym.cos(3*mi*z)/(8*mi**2) + z*sym.sin(mi*z)/(2*mi))
    psi21_p0ba_c = mc**2*sym.sin(mi*Hc)**2*sym.cos(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p0ab_t = mt**2*sym.cos(mi*Hi)**2*sym.sin(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p0ab_i = -mi**2*(-sym.sin(3*mi*z)/(8*mi**2) + z*sym.cos(mi*z)/(2*mi))
    psi21_p0ab_c = mc**2*sym.cos(mi*Hc)**2*sym.sin(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p0bb_t = mt**2*sym.cos(mi*Hi)**3/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p0bb_i = -mi**2*(-sym.cos(3*mi*z)/(8*mi**2) + z*sym.sin(mi*z)/(2*mi))
    psi21_p0bb_c = mc**2*sym.cos(mi*Hc)**3/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p0aab_t = mt**2*sym.sin(mi*Hi)**2*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p0aab_i = mi**2*(-sym.cos(3*mi*z)/(8*mi**2) - z*sym.sin(mi*z)/(2*mi))
    psi21_p0aab_c = mc**2*sym.sin(mi*Hc)**2*sym.cos(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p0bab_t = mt**2*sym.cos(mi*Hi)**2*sym.sin(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p0bab_i = mi**2*(sym.sin(3*mi*z)/(8*mi**2) + z*sym.cos(mi*z)/(2*mi))
    psi21_p0bab_c = mc**2*sym.cos(mi*Hc)**2*sym.sin(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    #12 for the 0-1 interaction
    psi21_p1aa_t = m0t/2*sym.sin(m0i*Hi)/sym.sin(m0t*(Hi-HT))*sym.sin(mi*Hi)/sym.sinh(mt*(Hi-HT))*(sym.sinh((mt+I*m0t)*(z-HT))/((mt+I*m0t)**2 - mt**2) + sym.sinh((mt-I*m0t)*(z-HT))/((mt-I*m0t)**2 - mt**2))
    psi21_p1aa_i = -m0i/2*(sym.sin((mi+m0i)*z)/((mi+m0i)**2 - mi**2) + sym.sin((mi-m0i)*z)/((mi-m0i)**2 - mi**2))
    psi21_p1aa_c = m0c/2*sym.sin(m0i*Hc)/sym.sin(m0c*Hc)*sym.sin(mi*Hc)/sym.sinh(mc*Hc)*(sym.sinh((mc+I*m0c)*z)/((mc+I*m0c)**2 - mc**2) + sym.sinh((mc-I*m0c)*z)/((mc-I*m0c)**2 - mc**2))

    psi21_p1ba_t = m0t/2*sym.sin(m0i*Hi)/sym.sin(m0t*(Hi-HT))*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))*(sym.sinh((mt+I*m0t)*(z-HT))/((mt+I*m0t)**2 - mt**2) + sym.sinh((mt-I*m0t)*(z-HT))/((mt-I*m0t)**2 - mt**2))
    psi21_p1ba_i = -m0i/2*(sym.cos((mi+m0i)*z)/((mi+m0i)**2 - mi**2) + sym.cos((mi-m0i)*z)/((mi-m0i)**2 - mi**2))
    psi21_p1ba_c = m0c/2*sym.sin(m0i*Hc)/sym.sin(m0c*Hc)*sym.cos(mi*Hc)/sym.sinh(mc*Hc)*(sym.sinh((mc+I*m0c)*z)/((mc+I*m0c)**2 - mc**2) + sym.sinh((mc-I*m0c)*z)/((mc-I*m0c)**2 - mc**2))

    psi21_p1ab_t = m0t/2*sym.cos(m0i*Hi)/sym.sin(m0t*(Hi-HT))*sym.sin(mi*Hi)/sym.sinh(mt*(Hi-HT))*(sym.sinh((mt+I*m0t)*(z-HT))/((mt+I*m0t)**2 - mt**2) + sym.sinh((mt-I*m0t)*(z-HT))/((mt-I*m0t)**2 - mt**2))
    psi21_p1ab_i = -m0i/2*(sym.cos((mi+m0i)*z)/((mi+m0i)**2 - mi**2) - sym.cos((mi-m0i)*z)/((mi-m0i)**2 - mi**2))
    psi21_p1ab_c = m0c/2*sym.cos(m0i*Hc)/sym.sin(m0c*Hc)*sym.sin(mi*Hc)/sym.sinh(mc*Hc)*(sym.sinh((mc+I*m0c)*z)/((mc+I*m0c)**2 - mc**2) + sym.sinh((mc-I*m0c)*z)/((mc-I*m0c)**2 - mc**2))

    psi21_p1bb_t = m0t/2*sym.cos(m0i*Hi)/sym.sin(m0t*(Hi-HT))*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))*(sym.sinh((mt+I*m0t)*(z-HT))/((mt+I*m0t)**2 - mt**2) + sym.sinh((mt-I*m0t)*(z-HT))/((mt-I*m0t)**2 - mt**2))
    psi21_p1bb_i = m0i/2*(sym.sin((mi+m0i)*z)/((mi+m0i)**2 - mi**2) - sym.sin((mi-m0i)*z)/((mi-m0i)**2 - mi**2))
    psi21_p1bb_c = m0c/2*sym.cos(m0i*Hc)/sym.sin(m0c*Hc)*sym.cos(mi*Hc)/sym.sinh(mc*Hc)*(sym.sinh((mc+I*m0c)*z)/((mc+I*m0c)**2 - mc**2) + sym.sinh((mc-I*m0c)*z)/((mc-I*m0c)**2 - mc**2))

    psi21_p1ac_t = m0t/2*(-J_psi10_pd_Hi + sym.sin(m0i*Hi))/sym.sin(m0t*(Hi-HT))*sym.sin(mi*Hi)/sym.sinh(mt*(Hi-HT))*(sym.sinh((mt+I*m0t)*(z-HT))/((mt+I*m0t)**2 - mt**2) + sym.sinh((mt-I*m0t)*(z-HT))/((mt-I*m0t)**2 - mt**2))
    psi21_p1ac_i = -m0i/2*(sym.sin((mi+m0i)*z)/((mi+m0i)**2 - mi**2) + sym.sin((mi-m0i)*z)/((mi-m0i)**2 - mi**2))
    psi21_p1ac_c = m0c/2*(J_psi10_pd_Hc + sym.sin(m0i*Hc))/sym.sin(m0c*Hc)*sym.sin(mi*Hc)/sym.sinh(mc*Hc)*(sym.sinh((mc+I*m0c)*z)/((mc+I*m0c)**2 - mc**2) + sym.sinh((mc-I*m0c)*z)/((mc-I*m0c)**2 - mc**2))

    psi21_p1bc_t = m0t/2*(-J_psi10_pd_Hi + sym.sin(m0i*Hi))/sym.sin(m0t*(Hi-HT))*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))*(sym.sinh((mt+I*m0t)*(z-HT))/((mt+I*m0t)**2 - mt**2) + sym.sinh((mt-I*m0t)*(z-HT))/((mt-I*m0t)**2 - mt**2))
    psi21_p1bc_i = -m0i/2*(sym.cos((mi+m0i)*z)/((mi+m0i)**2 - mi**2) + sym.cos((mi-m0i)*z)/((mi-m0i)**2 - mi**2))
    psi21_p1bc_c = m0c/2*(J_psi10_pd_Hc + sym.sin(m0i*Hc))/sym.sin(m0c*Hc)*sym.cos(mi*Hc)/sym.sinh(mc*Hc)*(sym.sinh((mc+I*m0c)*z)/((mc+I*m0c)**2 - mc**2) + sym.sinh((mc-I*m0c)*z)/((mc-I*m0c)**2 - mc**2))

    psi21_p1apa_t = mt**2/(m0t**2 + 4*mt**2)*sym.sin(mi*Hi)**3/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p1apa_i = mi**2/(m0i**2 - 4*mi**2)*(-sym.sin(3*mi*z)/(8*mi**2) + z*sym.cos(mi*z)/(2*mi))
    psi21_p1apa_c = mc**2/(m0c**2 + 4*mc**2)*sym.sin(mi*Hc)**3/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p1bpa_t = mt**2/(m0t**2 + 4*mt**2)*sym.sin(mi*Hi)**2*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p1bpa_i = mi**2/(m0i**2 - 4*mi**2)*(-sym.cos(3*mi*z)/(8*mi**2) + z*sym.sin(mi*z)/(2*mi))
    psi21_p1bpa_c = mc**2/(m0c**2 + 4*mc**2)*sym.sin(mi*Hc)**2*sym.cos(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p1apb_t = mt**2/(m0t**2 + 4*mt**2)*sym.sin(mi*Hi)*sym.cos(mi*Hi)**2/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p1apb_i = mi**2/(m0i**2 - 4*mi**2)*(sym.sin(3*mi*z)/(8*mi**2) - z*sym.cos(mi*z)/(2*mi))
    psi21_p1apb_c = mc**2/(m0c**2 + 4*mc**2)*sym.sin(mi*Hc)*sym.cos(mi*Hc)**2/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p1bpb_t = mt**2/(m0t**2 + 4*mt**2)*sym.cos(mi*Hi)**3/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p1bpb_i = mi**2/(m0i**2 - 4*mi**2)*(sym.cos(3*mi*z)/(8*mi**2) - z*sym.sin(mi*z)/(2*mi))
    psi21_p1bpb_c = mc**2/(m0c**2 + 4*mc**2)*sym.cos(mi*Hc)**3/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p1apab_t = mt**2/2/(m0t**2 + 4*mt**2)*sym.sin(2*mi*Hi)*sym.sin(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p1apab_i = mi**2/(m0i**2 - 4*mi**2)*(-sym.cos(3*mi*z)/(8*mi**2) - z*sym.sin(mi*z)/(2*mi))
    psi21_p1apab_c = mc**2/2/(m0c**2 + 4*mc**2)*sym.sin(2*mi*Hc)*sym.sin(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    psi21_p1bpab_t = mt**2/2/(m0t**2 + 4*mt**2)*sym.sin(2*mi*Hi)*sym.cos(mi*Hi)/sym.sinh(mt*(Hi-HT))**3*(sym.sinh(3*mt*(z-HT))/(8*mt**2) - z*sym.cosh(mt*(z-HT))/(2*mt))
    psi21_p1bpab_i = mi**2/(m0i**2 - 4*mi**2)*(sym.sin(3*mi*z)/(8*mi**2) + z*sym.cos(mi*z)/(2*mi))
    psi21_p1bpab_c = mc**2/2/(m0c**2 + 4*mc**2)*sym.sin(2*mi*Hc)*sym.cos(mi*Hc)/sym.sinh(mc*Hc)**3*(sym.sinh(3*mc*z)/(8*mc**2) - z*sym.cosh(mc*z)/(2*mc))

    #Full Paticular solution without correct continuity
    psi21_pd_t = N2t*(2*k**3/om**3 - k/om/om_p**2 - k**2/om**2/om_p)*(\
                (m0t**2*k/om*(1 + 2*k/om*om_p)*(1-k/om*om_p)*(v1[0]**3*psi21_p1apa_t + v1[0]**2*v1[1]*psi21_p1bpa_t + 2*v1[0]**2*v1[1]*psi21_p1apab_t + \
                                                            2*v1[0]*v1[1]**2*psi21_p1bpab_t + v1[0]*v1[1]**2*psi21_p1apb_t + v1[1]**3*psi21_p1bpb_t)) \
                + v1[0]*psi21_p1ac_t + v1[1]*psi21_p1bc_t + v1[0]*W10[0,0]*psi21_p1aa_t + v1[1]*W10[0,0]*psi21_p1ba_t + v1[0]*W10[1,0]*psi21_p1ab_t + v1[1]*W10[1,0]*psi21_p1bb_t) + \
                N2t*(k**2/om**2/om_p**2*(1+2*k/om*om_p)*(1-k/om*om_p) + k**3/om**3/om_p*(1-k/om*om_p))* \
                (v1[0]**3*psi21_p0aa_t + v1[0]**2*v1[1]*psi21_p0ba_t + 2*v1[0]**2*v1[1]*psi21_p0aab_t + 2*v1[0]*v1[1]**2*psi21_p0bab_t \
                + v1[0]*v1[1]**2*psi21_p0ab_t + v1[1]**3*psi21_p0bb_t)
    psi21_pd_i = N2i*(2*k**3/om**3 - k/om/om_p**2 - k**2/om**2/om_p)*(\
                (m0i**2*k/om*(1 + 2*k/om*om_p)*(1-k/om*om_p)*(v1[0]**3*psi21_p1apa_i + v1[0]**2*v1[1]*psi21_p1bpa_i + 2*v1[0]**2*v1[1]*psi21_p1apab_i + \
                                                            2*v1[0]*v1[1]**2*psi21_p1bpab_i + v1[0]*v1[1]**2*psi21_p1apb_i + v1[1]**3*psi21_p1bpb_i)) \
                + v1[0]*psi21_p1ac_i + v1[1]*psi21_p1bc_i + v1[0]*W10[0,0]*psi21_p1aa_i + v1[1]*W10[0,0]*psi21_p1ba_i + v1[0]*W10[1,0]*psi21_p1ab_i + v1[1]*W10[1,0]*psi21_p1bb_i) + \
                N2i*(k**2/om**2/om_p**2*(1+2*k/om*om_p)*(1-k/om*om_p) + k**3/om**3/om_p*(1-k/om*om_p))* \
                (v1[0]**3*psi21_p0aa_i + v1[0]**2*v1[1]*psi21_p0ba_i + 2*v1[0]**2*v1[1]*psi21_p0aab_i + 2*v1[0]*v1[1]**2*psi21_p0bab_i \
                + v1[0]*v1[1]**2*psi21_p0ab_i + v1[1]**3*psi21_p0bb_i)
    psi21_pd_c = N2c*(2*k**3/om**3 - k/om/om_p**2 - k**2/om**2/om_p)*(\
                (m0c**2*k/om*(1 + 2*k/om*om_p)*(1-k/om*om_p)*(v1[0]**3*psi21_p1apa_c + v1[0]**2*v1[1]*psi21_p1bpa_c + 2*v1[0]**2*v1[1]*psi21_p1apab_c + \
                                                            2*v1[0]*v1[1]**2*psi21_p1bpab_c + v1[0]*v1[1]**2*psi21_p1apb_c + v1[1]**3*psi21_p1bpb_c)) \
                + v1[0]*psi21_p1ac_c + v1[1]*psi21_p1bc_c + v1[0]*W10[0,0]*psi21_p1aa_c + v1[1]*W10[0,0]*psi21_p1ba_c + v1[0]*W10[1,0]*psi21_p1ab_c + v1[1]*W10[1,0]*psi21_p1bb_c) + \
                N2c*(k**2/om**2/om_p**2*(1+2*k/om*om_p)*(1-k/om*om_p) + k**3/om**3/om_p*(1-k/om*om_p))* \
                (v1[0]**3*psi21_p0aa_c + v1[0]**2*v1[1]*psi21_p0ba_c + 2*v1[0]**2*v1[1]*psi21_p0aab_c + 2*v1[0]*v1[1]**2*psi21_p0bab_c \
                + v1[0]*v1[1]**2*psi21_p0ab_c + v1[1]**3*psi21_p0bb_c)

    #Correct jumps at Hi and Hc
    J_psi21_pd_Hi = psi21_pd_t.subs({z:Hi}) - psi21_pd_i.subs({z:Hi})
    J_psi21_pd_Hc = psi21_pd_i.subs({z:Hc}) - psi21_pd_c.subs({z:Hc})

    psi21_p_t = psi21_pd_t + (J_psi21_Hi - J_psi21_pd_Hi + sym.sin(mi*Hi))*sym.sinh(mt*(z-HT))/sym.sinh(mt*(Hi-HT))
    psi21_p_i = psi21_pd_i + sym.sin(mi*z)
    psi21_p_c = psi21_pd_c + (-J_psi21_Hc + J_psi21_pd_Hc + sym.sin(mi*Hc))*sym.sinh(mc*z)/sym.sinh(mc*Hc)

    #Continuous pressure at Hi
    J_dpsi21_Hi = sym.diff(psi21_p_t, z).subs({z:Hi}) - sym.diff(psi21_p_i, z).subs({z:Hi})

    alpha_21 = ( (N2t - N2i)*(5/2*k**4/om**4*psih_0_i**2*sym.diff(psih_0_i, z) - 2*k**3/om**3*psih_0_i*psih_12_i + k**2/om**2*Zih_10*psih_0_i + \
                            k/om/om_p**2*psih_0_i*(k/om*(1+2*k/om*om_p)*(1-k/om*om_p)*sym.diff(psih_0_i**2, z) - psih_10_i)).subs({z:Hi})\
                - J_dpsi21_Hi )/J_dphi

    psih_21_i = psi21_p_i + alpha_21*phi_i
    psih_21_c = psi21_p_c + alpha_21*phi_c

    J_dpsi21_Hc = sym.diff(psih_21_i, z).subs({z:Hc}) - sym.diff(psih_21_c, z).subs({z:Hc})

    gamma = (1j/del_T)*( (N2i-N2c)*( -(1+L)*(5/2+2*L)*k**4/om**4*psih_0_i**2*sym.diff(psih_0_i, z) \
                                    + 2*(1+L)*k**3/om**3*psih_0_i*psih_12_i - k**2/om**2*Zch_10*psih_0_i \
                        - (1+L)*k/om/om_p**2*psih_0_i*(k/om*(1+2*k/om*om_p)*(1-k/om*om_p)*sym.diff(psih_0_i**2, z) - psih_10_i) ).subs({z:Hc}) \
                        + J_dpsi21_Hc)

    return gamma

# Compute the Det(M0) with M0 given by Eq (4.49). 
# This is needed to find the mean mode resonance wave numbers.
# This routine is a slimmed down version of the computation in nl_wave_sol.ipynb.
# See nl_wave_sol.ipynb for more detailed comments in code.
#
# inputs:
#       k_val  - numerical value of the horizontal wavenumber
#       om_val - numerical value of the frequency of the wave
# returns:
#       Det(M0)  - determinant of the matrix M0 given by Eq (4.49)
def make_detM0(k_val, om_val):
    a = mt*sym.sin(mi*Hi)/sym.tanh(mt*(Hi-HT)) - mi*sym.cos(mi*Hi)
    b = mt*sym.cos(mi*Hi)/sym.tanh(mt*(Hi-HT)) + mi*sym.sin(mi*Hi)

    c = mi*sym.cos(mi*Hc) - mc*sym.sin(mi*Hc)/sym.tanh(mc*Hc)
    d =-mi*sym.sin(mi*Hc) - mc*sym.cos(mi*Hc)/sym.tanh(mc*Hc)

    lambda2 = (1/2)*(sym.sqrt((a - d)**2 + 4*b*c) + a + d).subs({k:   k_val, om:  om_val})
    v2 = [
        (a - d + sym.sqrt((a - d)**2 + 4*b*c))/(2*c),
        1
    ]

    lambda1 = (1/2)*(-sym.sqrt((a - d)**2 + 4*b*c) + a + d).subs({k:   k_val, om:  om_val})
    v1 = [
        (a - d - sym.sqrt((a - d)**2 + 4*b*c))/(2*c),
        1
    ]

    if np.abs(lambda1) > np.abs(lambda2):
        
        temp = [v2[0], v2[1]]
        v2 = [v1[0], v1[1]]
        v1 = temp

    #Leading Order Solution
    psia_t = sym.sin(mi*Hi)*sym.sinh(mt*(z-HT))/sym.sinh(mt*(Hi-HT))
    psia_i = sym.sin(mi*z)
    psia_c = sym.sin(mi*Hc)*sym.sinh(mc*z)/sym.sinh(mc*Hc)

    psib_t = sym.cos(mi*Hi)*sym.sinh(mt*(z-HT))/sym.sinh(mt*(Hi-HT))
    psib_i = sym.cos(mi*z)
    psib_c = sym.cos(mi*Hc)*sym.sinh(mc*z)/sym.sinh(mc*Hc)

    psih_0_t = v1[0]*psia_t + v1[1]*psib_t
    psih_0_i = v1[0]*psia_i + v1[1]*psib_i
    psih_0_c = v1[0]*psia_c + v1[1]*psib_c

    #Build phi function
    phi_t = v2[0]*psia_t + v2[1]*psib_t
    phi_i = v2[0]*psia_i + v2[1]*psib_i
    phi_c = v2[0]*psia_c + v2[1]*psib_c

    #phi derivative jump at inversion
    J_dphi = sym_to_eval(sym.diff(phi_t,z), Hi, k_val, om_val)-sym_to_eval(sym.diff(phi_i,z), Hi, k_val, om_val)

    #print(f'Derivative jump in phi at inversion: {J_dphi}')



    #Compute psi^X, psi^T
    psih_X_t = -I*sym.diff(psih_0_t, k)
    psih_X_i = -I*sym.diff(psih_0_i, k)
    psih_X_c = -I*sym.diff(psih_0_c, k)

    psih_T_t =  I*sym.diff(psih_0_t, om)
    psih_T_i =  I*sym.diff(psih_0_i, om)
    psih_T_c =  I*sym.diff(psih_0_c, om)

    # Add phi to make derivative continuity at Hi
    del_X_Hi = sym_to_eval(sym.diff(psih_X_t,z), Hi, k_val, om_val)-sym_to_eval(sym.diff(psih_X_i,z), Hi, k_val, om_val)
    del_T_Hi = sym_to_eval(sym.diff(psih_T_t,z), Hi, k_val, om_val)-sym_to_eval(sym.diff(psih_T_i,z), Hi, k_val, om_val)

    alpha_X = -del_X_Hi/J_dphi
    alpha_T = -del_T_Hi/J_dphi

    psih_X_t += alpha_X*phi_t
    psih_X_i += alpha_X*phi_i
    psih_X_c += alpha_X*phi_c

    psih_T_t += alpha_T*phi_t
    psih_T_i += alpha_T*phi_i
    psih_T_c += alpha_T*phi_c

    del_X = sym_to_eval(sym.diff(psih_X_i,z), Hc, k_val, om_val)-sym_to_eval(sym.diff(psih_X_c,z), Hc, k_val, om_val)
    del_T = sym_to_eval(sym.diff(psih_T_i,z), Hc, k_val, om_val)-sym_to_eval(sym.diff(psih_T_c,z), Hc, k_val, om_val)

    om_p = del_X/del_T

    #print(f'group velocity = {om_p}')


    #Nonlinear terms (1,0)
    Nt = np.sqrt(N2t)
    Ni = np.sqrt(N2i)
    Nc = np.sqrt(N2c)

    m0t = Nt/om_p
    m0i = Ni/om_p
    m0c = Nc/om_p

    #Continuous homogeneous solutions
    psi10a_t = sym.sin(m0i*Hi)*sym.sin(m0t*(z-HT))/sym.sin(m0t*(Hi-HT))
    psi10a_i = sym.sin(m0i*z)
    psi10a_c = sym.sin(m0i*Hc)*sym.sin(m0c*z)/sym.sin(m0c*Hc)

    psi10b_t = sym.cos(m0i*Hi)*sym.sin(m0t*(z-HT))/sym.sin(m0t*(Hi-HT))
    psi10b_i = sym.cos(m0i*z)
    psi10b_c = sym.cos(m0i*Hc)*sym.sin(m0c*z)/sym.sin(m0c*Hc)

    #Build the solution with the correct derivative jump conditions
    J_dpsi10a_Hi = sym.diff(psi10a_t, z).subs({z:Hi}) - sym.diff(psi10a_i, z).subs({z:Hi})
    J_dpsi10b_Hi = sym.diff(psi10b_t, z).subs({z:Hi}) - sym.diff(psi10b_i, z).subs({z:Hi})
    J_dpsi10a_Hc = sym.diff(psi10a_i, z).subs({z:Hc}) - sym.diff(psi10a_c, z).subs({z:Hc})
    J_dpsi10b_Hc = sym.diff(psi10b_i, z).subs({z:Hc}) - sym.diff(psi10b_c, z).subs({z:Hc})\
    
    return (J_dpsi10a_Hi*J_dpsi10b_Hc - J_dpsi10b_Hi*J_dpsi10a_Hc).subs({k:k_val,om:om_val}).doit()



#################################################
#                                               #
#                   Plotting                    #
#                                               #
#################################################

# False generates the data this will take a while ~hour
# True load the data from a saved file - useful for editting the plot formatting
loadData = True

# Wavenumbers to plot
ks     = np.linspace(k_val, 2.5, num=201)
oms    = np.empty_like(ks)

for j in range(oms.size):
    if j == 0: oms[j] = om_val
    else: oms[j] = secant_omega(oms[j-1], N2t=N2t, N2i=N2i, N2c=N2c, Hc=Hc, Hi=Hi, HT=HT, k=ks[j])

Ls = np.linspace(0.0, 3.0, 201)

kk, LL = np.meshgrid(ks, Ls)
gams = np.empty_like(kk)

if loadData: gams = np.load('gamma_data.npy')

else:

    for idk, k_val in enumerate(ks):
        print(f'computing gamma for k = {k_val}')
        gamma = get_gamma(k_val, oms[idk]).subs({k:k_val, om:oms[idk]}).doit()
        print('evaluating at L values...')
        for idL, L_val in enumerate(Ls):
            gams[idL, idk] = sym_to_eval(gamma, Hc, L_val=Ls[idL], k_val=k_val, om_val=oms[idk]).real
        print()

    np.save('gamma_data.npy', gams)

k_guess_list  = np.array([2.0325,
                          2.0575,
                          2.1050,
                          2.1375,
                          2.1725,
                          2.2110,
                          2.2355,
                          2.2800,
                          2.3110,
                          2.3775,
                          2.4050,
                          2.4700,
                          2.4950,
                          2.4370,
                          2.3357])
om_guess_list = np.array([7.98195,
                          7.99439,
                          8.01732,
                          8.03249,
                          8.04838,
                          8.06534,
                          8.07587,
                          8.09447,
                          8.10705,
                          8.13304,
                          8.14340,
                          8.16707,
                          8.17587,
                          8.15520,
                          8.11686])

k_resonance = []

#Secant method to find 0's of det(M0) (mean mode resonances)
for k_guess, om_guess in zip(k_guess_list, om_guess_list):
    print(f'Computing resonance wavenumber near {k_guess}')
    k_1 = k_guess
    k_2 = k_guess*(1.0001)
    while(np.abs((k_1-k_2)/k_1)>1e-5):
        temp = k_1
        DetM0_1 = float(make_detM0(k_val=k_1, om_val=secant_omega(om_guess, N2t=N2t, N2i=N2i, N2c=N2c, 
                                                            Hc=Hc, Hi=Hi, HT=HT, k=k_1)))
        DetM0_2 = float(make_detM0(k_val=k_2, om_val=secant_omega(om_guess, N2t=N2t, N2i=N2i, N2c=N2c, 
                                                            Hc=Hc, Hi=Hi, HT=HT, k=k_2)))
        k_1 = k_1 - DetM0_1*(k_1-k_2)/(DetM0_1-DetM0_2)
        k_2 = temp
    k_resonance.append(k_1)


#Plot thesis version (Figure 4.6)
fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(8,3)
fig.set_dpi(500)

gammax = 2.5
gam_round = np.maximum(gams, -gammax)
gam_round = np.minimum(gam_round, gammax)

contf = ax.contourf(kk, LL, gam_round, levels=np.linspace(-gammax, gammax, 21), cmap='PRGn')
ax.contour(kk, LL, gams, levels=[0], colors='k', linewidths=0.5)
fig.colorbar(contf, label=r'$\gamma$')
ax.set(title=r'Nonlinear NLS Coefficient', 
       ylabel=r'Latent Heat Forcing $\mathcal{L}$', 
       xlabel=r'Horizontal Wavenumber $k$')
for k_res in k_resonance:
    ax.axvline(k_res, color='brown', linestyle='--', linewidth=1.25)

fig.savefig('./nonlinear_coef.png', dpi=500)


#Plot presentation version
fig2, ax2 = plt.subplots(layout='constrained')
fig2.set_size_inches(8,3)
fig2.set_dpi(500)

contf2 = ax2.contourf(kk, LL, gam_round, levels=np.linspace(-gammax, gammax, 21), cmap='PRGn')
ax2.contour(kk, LL, gams, levels=[0], colors='k', linewidths=0.5)
fig2.colorbar(contf2)
ax2.set(title=r'Nonlinear Coefficient', 
       ylabel=r'Latent Heat Forcing $\mathcal{L}$', 
       xlabel=r'Horizontal Wavenumber $k$')
for k_res in k_resonance:
    ax2.axvline(k_res, color='brown', linestyle='--', linewidth=1.25)

#This shows up in the correct spot with savefig not with plt.show()
ax2.annotate(
        r'$\gamma$',
        xy=(0.63, 1.04),  # Position relative to axes (outside)
        xycoords='axes fraction',
        color='darkcyan',
        size=14
    )

fig2.savefig('./nonlinear_coef_pres.png', dpi=500)

plt.close()
