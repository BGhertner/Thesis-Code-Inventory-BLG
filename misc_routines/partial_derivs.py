#
# Ben Ghertner 2025
#
# Generate partial derivative coefs by 
# solving eq. (A.4) and eq. (A.12) (cloud)
#
#
###               WARNING                ###
#
# this routine has not be thoroughly checked 
# and debugged use at your own risk!

import sympy as sym

def get_partial_derivatives():
    """
    Symbolically solve the matrix system (A.4) and (A.12) for the partial derivative coefs.

    Returns:

    Ju - (2x3 sympy matrix) The unsaturated Jacobian (A.5) .

    Js - (3x3 sympy matrix) The saturated Jacobian.

    (T, rl, p, rT) - (tuple of sympy symbols) Sympy symbol for the thermodynamic variables
                     used in Ju and Js.

    """

    #variables
    T, rl, p, rT = sym.symbols('T r_l p r_T')

    #constants
    Rd  = 287.0
    Rv  = 461.5
    cpd = 1006.
    cpv = 1870.
    cpl = 4190.
    p0  = 1e5

    #Functions of variables
    lv = 2.501e6 + (cpl - cpv)*(T - 273.15)
    RT = Rd + rT*Rv
    cpT=cpd + rT*cpv
    rs = rT - rl        #(in the cloud)
    Rs = Rd + rs*Rv
    lambda_c = lv/cpT/T
    lambda_R = lv/Rv/T
    M = RT*sym.log(RT/Rs) + rT*Rv*sym.log(rs/rT)

    #Derivative functions
    dlam_c_dT  = sym.diff(lambda_c, T)
    dlam_c_drT = sym.diff(lambda_c, rT)
    dM_drT     = sym.diff(M, rT)
    dM_drl     = sym.diff(M, rl)
    dRTcpT_drT = sym.diff(RT/cpT, rT)

    #Clear Matrices
    Au = sym.Matrix([[1, -1],
                     [0,  1]])
    Bu = sym.Matrix([[0,      0, (Rv-Rd)/RT*rT/(1+rT)],
                     [RT/cpT, 1, dRTcpT_drT*rT*sym.log(p/p0)]])
    
    #Clear Jacobian
    Ju = Au**(-1)@Bu

    #Saturated Matrices
    a22 = 1 - T*dlam_c_dT*rl
    a23 = -lambda_c - 1/cpT*dM_drl

    As = sym.Matrix([[1, -1,                Rv/Rs],
                     [0, a22,               a23  ],
                     [0, lambda_R*Rs/Rd*rs, 1]])
    
    b23 = (dRTcpT_drT*sym.log(p/p0) + dlam_c_drT*rl - cpv/cpT**2*M + 1/cpT*dM_drT)*rT

    Bs = sym.Matrix([[0,        0, (Rv-Rd)/Rs*rT/(1+rT)],
                     [RT/cpT,   1, b23],
                     [Rs/Rd*rs, 0, rT]])
    
    #Saturated Jacobian
    Js = As**(-1)@Bs

    return Ju, Js, (T, rl, p, rT)
        
