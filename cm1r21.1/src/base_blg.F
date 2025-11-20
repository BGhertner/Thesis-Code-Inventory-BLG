MODULE base_blg_module

implicit none

private
public :: inversionfog_base, mixedcldlayer_base, mixedunsatlayer_base 
public :: mixeddrylayer_base, marginalcld_base, cldoverclear_base
public :: linearshear_wind

CONTAINS

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Start definition of base state thermo (isnd option)  ccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


!-----------------------------------------------------------------------------
!
!  subroutine inversionfog_base:  Specify base-state arrays 
!                              for inversion fog set-up.
!  &param8 inputs:
!            var1 : surface pressure [Pa]
!            var2 : surface temperature [K]
!            var3 : inversion N^2 [1/s^2]
!            var4 : troposphere N^2 [1/s^2]
!            var5 : inversion height [m]
!            var6 : fog top height [m]
!            var7 : liquid water slope at cloud edge [1/m]
!            var8 : far field relative humidity
!            var9 : water smoothing parameter
!
!  NOTES: BLG 14 Aug 2024
!         -Piecewise Constant d/dz log(thr) inversion layer
!         -Lower rT function has linear relative humidity
!         -Upper rT has constant relative humidity as z -> infty
!         -Two stage approach similiar to the Walsh and Muraki 2024
!         -Second stage uses Newton's method for T, rT root finding
!          to fix unstable fixed point iteration of RPW for warm temps 
!
!-----------------------------------------------------------------------------

subroutine inversionfog_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants
    use cm1libs , only : rslf

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0
    real, dimension(ib:ie,jb:je,kb:ke) :: thr0, thl0, rst0, qt0

    real, dimension(kb:ke) :: pistT, prstT, thvstT, rstT, TstT
    real, dimension(kb:ke) :: pistB, prstB, thvstB, rstB, TstB

    integer i,j,k, kbot, ktop

    real :: psfc, tsfc, pisfc, thsfc, qvsfc, thrsfc
    real :: nsbot, nstop, rhslope, hinf, gamma, Hi, Hc, lv_T, lambc
    real :: Tlast, picld, thrcld, H, rstTc, rstBc

    print *, 'Generating Base State for Inversion Fog'

    ! Save user input variables (set in &param8 in namelist.input)
    psfc = var1
    tsfc = var2

    nsbot = var3
    nstop = var4

    Hi = var5
    Hc = var6

    rhslope = var7
    hinf = var8
    gamma = var9

    ! Calculate other surface values
    pisfc = (psfc/p00)**(rd/cp)
    thsfc = Tsfc/pisfc
    qvsfc = rslf(psfc, tsfc)
    thrsfc = thsfc*(1.0 + reps*qvsfc)/(1.0 + qvsfc)

    !!STAGE 1 - make * values!!
    do k=1,ke
    
        !!!!theta_v and p are exact from specified (approx) N^2 unsaturated
        !Bottom
        thvstB(k) = thrsfc*exp(nsbot/g*zh(ib,jb,k))
        pistB(k) = pisfc - g**2/cp/thrsfc/nsbot*(1 - exp(-nsbot/g*zh(ib,jb,k)))
        prstB(k) = p00*(pistB(k)**(cp/rd))
        !Top
        thvstT(k) = thrsfc*exp((nsbot - nstop)/g*Hi + nstop/g*zh(ib,jb,k))
        pistT(k) = pisfc - g**2/cp/thrsfc*(1/nsbot + (1/nstop-1/nsbot)*exp(-nsbot/g*Hi) &
                                            -1/nstop*exp((nstop - nsbot)/g*Hi - nstop/g*zh(ib,jb,k)))
        prstT(k) = p00*(pistT(k)**(cp/rd))

        !!!!Nonlinear root finding for r* and T* in top and bottom
        !Bottom first
        Tlast = 0
        !Initial guess at T
        TstB(k) = thvstB(k)*pistB(k)

        do while((abs(Tlast-TstB(k))/TstB(k)).ge.1e-6)
            Tlast = TstB(k)
            !!! calc new r* and T from thr
            rstB(k) = rslf(prstB(k),TstB(k))
            ! update T
            TstB(k) = thvstB(k)*pistB(k)/(1.0 + reps*rstB(k))*(1.0 + rstB(k))
        enddo

        !Same for Top
        Tlast = 0
        !Initial guess at T
        TstT(k) = thvstT(k)*pistT(k)

        do while((abs(Tlast-TstT(k))/TstT(k)).ge.1e-6)
            Tlast = TstT(k)
            !!! calc new r* and T from thr
            rstT(k) = rslf(prstT(k),TstT(k))
            ! update T
            TstT(k) = thvstT(k)*pistT(k)/(1.0 + reps*rstT(k))*(1.0 + rstT(k))
        enddo

        !initialize total thermodynamic fields
        do i=ib,ie
        do j=jb,je

            IF(zh(i,j,k).le.Hi)THEN
                pi0(i,j,k)  = pistB(k)
                prs0(i,j,k) = prstB(k)
                T0(i,j,k)   = TstB(k)
                thr0(i,j,k) = thvstB(k)
                rst0(i,j,k) = rstB(k)
            ELSE
                pi0(i,j,k)  = pistT(k)
                prs0(i,j,k) = prstT(k)
                T0(i,j,k)   = TstT(k)
                thr0(i,j,k) = thvstT(k)
                rst0(i,j,k) = rstT(k)
            ENDIF

            !dry theta for CM1
            th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)
            !background thl
            thl0(i,j,k) = thl_func(prs0(i,j,k),T0(i,j,k),rst0(i,j,k))
            
        enddo
        enddo

    enddo

    !!STAGE 2 - add water, keep thl the same, update pressure!!

    !first find cloud edge!
    k = 1
    do while(zh(ib,jb,k).lt.Hc)
        k = k+1
    enddo

    !Cld edge is a grid point
    IF(zh(ib,jb,k).eq.Hc)THEN
        kbot=k-1
        ktop=k+1
        picld=pi0(ib,jb,k)
        thrcld=thr0(ib,jb,k)
        rstBc=rstB(k)
        rstTc=rstT(k)
    !Cld edge is between grid points
    ELSE
        kbot=k-1
        ktop=k
        !linear interp.
        picld=(pi0(ib,jb,kbot)*(Hc-zh(ib,jb,ktop)) + pi0(ib,jb,ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
        thrcld=(thr0(ib,jb,kbot)*(Hc-zh(ib,jb,ktop)) + thr0(ib,jb,ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
        rstBc=(rstB(kbot)*(Hc-zh(ib,jb,ktop)) + rstB(ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
        rstTc=(rstT(kbot)*(Hc-zh(ib,jb,ktop)) + rstT(ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
    ENDIF

    !Calculate H parameter for smooth total water profile
    H = (rstTc*hinf*(1+tanh(gamma*(Hc-Hi))) - rstBc*(1 + rhslope*Hc + (1 - rhslope*Hc)*tanh(gamma*(Hc-Hi)))) &
            /(rhslope*(tanh(gamma*(Hc-Hi)) - 1)*rstBc)

    do j=jb,je
    do i=ib,ie

    !start at cloud edge and go up (clear)
    do k=ktop,ke
        !New water
        qv0(i,j,k) = ( rstB(k)*( 1-rhslope*(zh(i,j,k)-H) ) + rstT(k)*hinf &
                        - ( rstB(k)*(1-rhslope*(zh(i,j,k)-H) ) - rstT(k)*hinf)*tanh(gamma*(zh(i,j,k)-Hi)) )/2
        qc0(i,j,k) = 0.0
        
        Tlast = 0.0
        
        !Nonlinear solve for pressure 
        !T and th are exact in clear when p and thl are known
        do while(abs((Tlast-th0(i,j,k))/th0(i,j,k)).gt.1e-7)
            Tlast = th0(i,j,k)
            th0(i,j,k) = thl0(i,j,k)*(p00/prs0(i,j,k))**&
                        ((rd*cpv-rv*cp)/(cp*(cp+qv0(i,j,k)*cpv))*qv0(i,j,k))
            thr0(i,j,k) = th0(i,j,k)*(1.0+reps*qv0(i,j,k))/(1.0+qv0(i,j,k))

            IF(k.eq.ktop)THEN
                pi0(i,j,k) = picld-g*(zh(i,j,k)-Hc)/(cp*0.5*(thr0(i,j,k)+thrcld))
            ELSE
                pi0(i,j,k) = pi0(i,j,k-1)-g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(thr0(i,j,k)+thr0(i,j,k-1)))
            ENDIF

            prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))
        enddo
        T0(i,j,k)=th0(i,j,k)*pi0(i,j,k)
        
    enddo

    !start at cloud edge and go down (cloud)
    !For sufficiently hot temperature fixed point method of RPW fails
    !Use an approximate newton's method
    do k=kbot,1,-1
        qt0(i,j,k) = ( rstB(k)*( 1-rhslope*(zh(i,j,k)-H) ) + rstT(k)*hinf &
                        - ( rstB(k)*( 1-rhslope*(zh(i,j,k)-H) ) - rstT(k)*hinf)*tanh(gamma*(zh(i,j,k)-Hi)) )/2

        !First guess using the liq. water function and linear T eq.
        lv_T = lv1 - lv2*T0(i,j,k)
        lambc = lv_T/(cp+qt0(i,j,k)*cpv)/T0(i,j,k)

        qc0(i,j,k) = (qt0(i,j,k) - rstB(k))*dlT_dlthl(prs0(i,j,k),T0(i,j,k))
        qv0(i,j,k) = qt0(i,j,k) - qc0(i,j,k)

        T0(i,j,k) = T0(i,j,k)*(1 + lambc*qc0(i,j,k))
        Tlast = 0.0

        !Nonlinear solve for pressure
        !Use Newton step to get T at each iteration
        do while(abs((Tlast-T0(i,j,k))/T0(i,j,k)).gt.3e-7)
            Tlast = T0(i,j,k)

            !Newton step
            T0(i,j,k) = T0(i,j,k)*(1 - log(thl_func(prs0(i,j,k),T0(i,j,k),qt0(i,j,k))/thl0(i,j,k))&
                                        *dlT_dlthl(prs0(i,j,k),T0(i,j,k)) )

            qv0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))
            qc0(i,j,k) = qt0(i,j,k) - qv0(i,j,k)

            th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)
            thr0(i,j,k) = th0(i,j,k)*(1.0+reps*qv0(i,j,k))/(1.0+qt0(i,j,k))

            IF(k.eq.ktop)THEN
                pi0(i,j,k) = picld-g*(zh(i,j,k)-Hc)/(cp*0.5*(thr0(i,j,k)+thrcld))
            ELSE
                pi0(i,j,k) = pi0(i,j,k+1)-g*(zh(i,j,k)-zh(i,j,k+1))/(cp*0.5*(thr0(i,j,k)+thr0(i,j,k+1)))
            ENDIF

            prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))

        enddo

    enddo

    enddo
    enddo

    return

    CONTAINS

    real function dlT_dlthl(prs,temp)
        real :: prs, temp, lamb, lambR, lambc, rtot, cptot, rst, lv_T
        
        lv_T  = lv1 - lv2*temp
        rst   = rslf(prs, temp)
        rtot  = rd + rst*rv
        cptot = cp + rst*cpv
        lambR = lv_T/rv/temp
        lambc = lv_T/cptot/temp
        lamb  = lambc*lambR*rtot/rd*rst

        dlT_dlthl = 1/(1+lamb)
    
    end function dlT_dlthl

    real function thl_func(prs,temp,qt)
        real :: prs, temp, qt, lambc, rtot, cptot, rst, lv_T, qv, qc

        lv_T  = lv1 - lv2*temp

        rst   = rslf(prs, temp)
        IF(rst.lt.qt)THEN
            qv = rst
            qc = qt-rst
        ELSE
            qv = qt
            qc = 0
        ENDIF

        rtot  = rd + qt*rv
        cptot = cp + qt*cpv
        lambc = lv_T/cptot/temp

        thl_func = temp*(p00/prs)**(rtot/cptot)*exp(-lambc*qc)&
                  *((rd+qv*rv)/rtot)**(rtot/cptot)*(qt/qv)**(qt*rv/cptot)

    end function thl_func

end subroutine inversionfog_base

!-----------------------------------------------------------------------------
!
!  subroutine mixedcldlayer_base:  Specify base-state arrays for 
!                                  mixed cloud layer aloft.
!  &param8 inputs :
!            var1 : Surface pressure [Pa]
!            var2 : Surface temperature [K]
!            var3 : Constant N^2 [1/s^2]
!            var4 : Height at center of layer [m]
!            var5 : Half height of layer [m]
!            var6 : 
!            var7 : 
!            var8 : Generalized relative humidity max (assumed > 1)
!            var9 : Far field (z -> +/- infty) relative humidity (assumed < 1)
!
!  NOTES: BLG 06 June 2025
!
!-----------------------------------------------------------------------------

subroutine mixedcldlayer_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants
    use cm1libs , only : rslf

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0
    real, dimension(ib:ie,jb:je,kb:ke) :: thr0, thl0, rst0, qt0,Tr0

    integer :: i,j,k,n

    real :: psfc, pisfc, Tsfc, thsfc, qvsfc, thrsfc, rstsfc, rTsfc, Trsfc
    real :: ns, Hcenter, Hdepth, hmax, hinf
    real :: Tlast, thrcld, Trcld, picld, Tcld, prscld, lv_T

    print *, 'Generating Base State for Well mixed cloud aloft'

    !define surface conditions
    psfc = var1
    pisfc = (psfc/p00)**(rd/cp)
    Tsfc = var2

    ! Define Brunt-Vaisala frequency N^2
    ns =  var3

    ! Neutral Layer parameters
    Hcenter = var4
    Hdepth = var5

    hmax = var8
    Hinf = var9

    ! Other surface values
    rstsfc = rslf(psfc, Tsfc)
    rTsfc = rstsfc*rhfunc(0.0)
    Trsfc = Tsfc*(rd + rTsfc*rv)/rd/(1 + rTsfc)
    thrsfc = Tsfc/pisfc*(rd+rTsfc*rv)/(1+rTsfc)/rd

    do i=ib,ie
    do j=jb,je
    do k=1,ke

        Tlast = 0.0
        n = 0

        ! Clear Regions integrate thr exactly in the clear regions
        if (zh(i,j,k).le.(Hcenter-Hdepth) .or.  zh(i,j,k).ge.(Hcenter+Hdepth))then
        
            if (zh(i,j,k).le.(Hcenter-Hdepth)) then
                thr0(i,j,k) = thrsfc*exp(ns/g*zh(i,j,k))
            else
                thr0(i,j,k) = thrcld*exp(ns/g*(zh(i,j,k) - (Hcenter+Hdepth)))
            endif

            ! Discretely step pressure forward
            if (k.eq.1) then 
                pi0(i,j,k)= pisfc - g*zh(i,j,k)/(cp*0.5*(thr0(i,j,k)+thrsfc))
            else 
                pi0(i,j,k)= pi0(i,j,k-1) - g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(thr0(i,j,k)+thr0(i,j,k-1)))
            endif
            
            prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))

            !Nonlinear root finding for theta_l and r*
            Tlast = 0
            n=1
            !Initial guess at T
            T0(i,j,k) = thr0(i,j,k)*pi0(i,j,k)

            do while((abs(Tlast-T0(i,j,k))/T0(i,j,k)).ge.1e-6)
                n=n+1
                !Check for infinite loop
                if (n.gt.100.0) stop 1111
                Tlast = T0(i,j,k)
                !!! calc new r* and T from thr
                rst0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))
                qv0(i,j,k)  = rst0(i,j,k)*rhfunc(zh(i,j,k))
                ! update T
                T0(i,j,k) = thr0(i,j,k)*pi0(i,j,k)/(1.0 + reps*qv0(i,j,k))*(1.0 + qv0(i,j,k))
            enddo

            qc0(i,j,k) = 0.0
            Tr0(i,j,k) = thr0(i,j,k)*pi0(i,j,k)

        !Cloudy Region numerically integrate for Tr
        else

            !Initialize at next grid point with previous grid point vals
            prs0(i,j,k) = prs0(i,j,k-1)
            T0(i,j,k)   = T0(i,j,k-1)
            rst0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))
            qv0(i,j,k)  = rst0(i,j,k)
            qc0(i,j,k)  = rst0(i,j,k)*(rhfunc(zh(i,j,k)) - 1)

            !First step into cloud compute cloud edge values
            if(zh(i,j,k-1).le.(Hcenter-Hdepth))then
                thrcld = thrsfc*exp(ns/g*(Hcenter-Hdepth))
                picld  = pi0(i,j,k-1) - g*(Hcenter-Hdepth-zh(i,j,k-1))/(cp*0.5*(thrcld+thr0(i,j,k-1)))
                prscld=p00*(picld**(cp/rd))
                Trcld = thrcld*picld

                !Nonlinear root finding for T and r*
                Tlast = 0
                !Initial guess at T
                Tcld = thrcld*picld
                do while((abs(Tlast-Tcld)/Tcld).ge.1e-6)
                    Tlast = Tcld
                    ! update T
                    Tcld = thrcld*picld/(1.0 + reps*rslf(prscld,Tcld))*(1.0 + rslf(prscld,Tcld))
                enddo

            endif

            n = 0
            Tlast = 0.0
            do while(abs((Tlast-T0(i,j,k))/T0(i,j,k)).gt.1e-6)
                n = n + 1
                Tlast = T0(i,j,k)
                ! Step forward Trho ! first step is special
                if(zh(i,j,k-1).le.(Hcenter-Hdepth))then
                    Tr0(i,j,k) = Trcld - g/rd*dlTr_dlp(prscld, Tcld)*(zh(i,j,k)-(Hcenter-Hdepth))
                else
                    Tr0(i,j,k) = Tr0(i,j,k-1) - g/rd*dlTr_dlp((prs0(i,j,k)+prs0(i,j,k-1))/2, &
                                                              (T0(i,j,k)+T0(i,j,k-1))/2)*(zh(i,j,k)-zh(i,j,k-1))
                endif
                !Compute thr (uses last iterations pi)
                thr0(i,j,k) = Tr0(i,j,k)/pi0(i,j,k)
                T0(i,j,k) = Tr0(i,j,k)/(1.0 + reps*qv0(i,j,k))*(1.0 + qv0(i,j,k) + qc0(i,j,k))
                ! Step forward pi and set pressure
                pi0(i,j,k)  = pi0(i,j,k-1) - g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(thr0(i,j,k)+thr0(i,j,k-1)))
                prs0(i,j,k) = p00*(pi0(i,j,k)**(cp/rd))
                ! compute water
                rst0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))
                qv0(i,j,k)  = rst0(i,j,k)
                qc0(i,j,k)  = rst0(i,j,k)*(rhfunc(zh(i,j,k)) - 1)
                if(n.gt.100.0) stop 1112
            enddo

            !Last step find thrcld value with simple forward Euler
            if (zh(i,j,k+1).ge.(Hcenter+Hdepth))then
                Trcld = Tr0(i,j,k) - g/rd*dlTr_dlp(prs0(i,j,k),T0(i,j,k))*(Hcenter+Hdepth-zh(i,j,k))
                picld = pi0(i,j,k) - g*(Hcenter+Hdepth-zh(i,j,k))/(cp*thr0(i,j,k))
                thrcld = Trcld/picld
            endif


        endif

        !dry theta for CM1
        th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)

        qt0(i,j,k) = qv0(i,j,k) + qc0(i,j,k)
        !background thl
        thl0(i,j,k) = thl_func(prs0(i,j,k),T0(i,j,k),qt0(i,j,k))

    enddo
    enddo
    enddo

    return

    CONTAINS

    real function rhfunc(z)
        real :: z
        
        rhfunc = (hmax - Hinf)*exp(1/Hdepth**2*log((1-Hinf)/(hmax - Hinf))*(z - Hcenter)**2) + Hinf
    
    end function rhfunc

    real function dlTr_dlp(prs,temp)
        real :: prs, temp, lamb, lambR, lambc, rtot, cptot, rst, lv_T
        
        lv_T  = lv1 - lv2*temp
        rst   = rslf(prs, temp)
        rtot  = rd + rst*rv
        cptot = cp + rst*cpv
        lambR = lv_T/rv/temp
        lambc = lv_T/cptot/temp
        lamb  = lambc*lambR*rtot/rd*rst

        dlTr_dlp = rtot/cptot - 1/(1+lamb)*rv/rd*rst*(rtot/cptot*lambR - 1)**2
    
    end function dlTr_dlp

    real function thl_func(prs,temp,qt)
        real :: prs, temp, qt, lambc, rtot, cptot, rst, lv_T, qv, qc

        lv_T  = lv1 - lv2*temp

        rst   = rslf(prs, temp)
        IF(rst.lt.qt)THEN
            qv = rst
            qc = qt-rst
        ELSE
            qv = qt
            qc = 0
        ENDIF

        rtot  = rd + qt*rv
        cptot = cp + qt*cpv
        lambc = lv_T/cptot/temp

        thl_func = temp*(p00/prs)**(rtot/cptot)*exp(-lambc*qc)&
                  *((rd+qv*rv)/rtot)**(rtot/cptot)*(qt/qv)**(qt*rv/cptot)

    end function thl_func

end subroutine mixedcldlayer_base

!-----------------------------------------------------------------------------
!
!  subroutine mixedunsatlayer_base:  Specify base-state arrays for 
!                                     mixed unsaturated layer aloft.
!  &param8 inputs:
!            var1 : Surface pressure [Pa]
!            var2 : Surface temperature [K]
!            var3 : Constant N^2 [1/s^2]
!            var4 : Height at center of layer [m]
!            var5 : Half height of layer [m]
!            var6 : 
!            var7 : 
!            var8 : 
!            var9 : Constant relative humidity
!
!  NOTES: BLG 07 Jan 2025
!         -Well mixed layer (N^2=0) from Hcenter - Hdepth ->  Hcenter + Hdepth
!         -Constant N^2 ( d/dz log(thr) ) above and below
!         -Constant relative humidity everywhere (assumed to be less than 1!)
!
!-----------------------------------------------------------------------------

subroutine mixedunsatlayer_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants
    use cm1libs , only : rslf

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0
    real, dimension(ib:ie,jb:je,kb:ke) :: thv0, thl0, qt0, rst0

    real :: psfc, pisfc, Tsfc, thsfc, qvsfc, thvsfc
    real :: ns, Hcenter, Hdepth, rhval, Tlast

    integer i,j,k,n

    print *, 'Generating Base State for Well mixed unsaturated layer aloft'
    
    !define surface conditions
    psfc = var1
    pisfc = (psfc/p00)**(rd/cp)
    Tsfc = var2

    ! Define Brunt-Vaisala frequency N^2
    ns =  var3

    ! Neutral Layer parameters
    Hcenter = var4
    Hdepth = var5

    ! Constant relative humidity
    rhval = var9

    ! Calculate other surface values
    thsfc = Tsfc/pisfc
    qvsfc = rslf(psfc, tsfc)*rhval
    thvsfc = thsfc*(1.0 + reps*qvsfc)/(1.0 + qvsfc)
    
    if(dowr) write(outfile,*)
    do j=jb,je
    do i=ib,ie
    do k=1,ke
        ! DEFINE \theta_v here for piecewise constant N^2 (unsaturated)
        if (zh(i,j,k).le.(Hcenter-Hdepth)) then
            thv0(i,j,k) = thvsfc*exp(ns/g*zh(i,j,k))
        elseif (zh(i,j,k).le.(Hcenter+Hdepth)) then
            thv0(i,j,k) = thvsfc*exp(ns/g*(Hcenter-Hdepth))
        else
            thv0(i,j,k) = thvsfc*exp(ns/g*(zh(i,j,k) - 2*Hdepth))
        endif

        ! Discretely step pressure forward
        if (k.eq.1) then 
            pi0(i,j,k)= pisfc - g*zh(i,j,k)/(cp*0.5*(thv0(i,j,k)+thvsfc))
        else 
            pi0(i,j,k)= pi0(i,j,k-1) - g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(thv0(i,j,k)+thv0(i,j,k-1)))
        endif
        
        prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))

    !Nonlinear root finding for theta_l and r*
    Tlast = 0
    n=1
    !Initial guess at T
    T0(i,j,k) = thv0(i,j,k)*pi0(i,j,k)

    do while((abs(Tlast-T0(i,j,k))/T0(i,j,k)).ge.1e-6)
        n=n+1
        !Check for infinite loop
        if (n.gt.100.0) stop 1111
        Tlast = T0(i,j,k)
        !!! calc new r* and T from thr
        qv0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))*rhval
        ! update T
        T0(i,j,k) = thv0(i,j,k)*pi0(i,j,k)/(1.0 + reps*qv0(i,j,k))*(1.0 + qv0(i,j,k))
    enddo

    !dry theta for CM1
    th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)

    !background thl
    thl0(i,j,k) = T0(i,j,k)*(p00/prs0(i,j,k))**((rd+qv0(i,j,k)*rv)/(cp+qv0(i,j,k)*cpv))

    enddo  ! enddo for k loop
    enddo  ! enddo for i loop
    enddo  ! enddo for j loop

    
    return
end subroutine mixedunsatlayer_base

!-----------------------------------------------------------------------------
!
!  subroutine mixeddrylayer_base:  Specify base-state arrays for 
!                                   mixed layer aloft dry everywhere.
!  &param8 inputs :
!            var1 : Surface pressure [Pa]
!            var2 : Surface temperature [K]
!            var3 : Constant N^2 [1/s^2]
!            var4 : Height at center of layer [m]
!            var5 : Half height of layer [m]
!            var6 : 
!            var7 : 
!            var8 : 
!            var9 :
!
!  NOTES: BLG Jan 2025
!         -Well mixed layer (N^2=0) from Hcenter - Hdepth ->  Hcenter + Hdepth
!         -Constant N^2 above and below
!
!-----------------------------------------------------------------------------

subroutine mixeddrylayer_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0,thl0,rst0

    real :: psfc, Tsfc, pisfc, thsfc, ns, Hcenter, Hdepth

    integer :: i,j,k

    print *, 'Generating Base State for Well mixed dry layer aloft'

    ! Define Surface conditions
    psfc = var1
    Tsfc = var2

    ! Define Brunt-Vaisala frequency N^2
    ns =  var3

    Hcenter = var4
    Hdepth = var5

    pisfc = (psfc/p00)**(rd/cp)
    thsfc = Tsfc/pisfc

    do k=kb,ke
    do j=jb,je
    do i=ib,ie

        if(zh(i,j,k).le.(Hcenter-Hdepth))then
            th0(i,j,k) = thsfc*exp(ns/g*zh(i,j,k))
        elseif(zh(i,j,k).le.(Hcenter+Hdepth))then
            th0(i,j,k) = thsfc*exp(ns/g*(Hcenter-Hdepth))
        else
            th0(i,j,k) = thsfc*exp(ns/g*(zh(i,j,k)-2*Hdepth))
        endif

        ! Discretely step pressure forward
        if (k.eq.1) then 
            pi0(i,j,k)= pisfc - g*zh(i,j,k)/(cp*0.5*(th0(i,j,k)+thsfc))
        else 
            pi0(i,j,k)= pi0(i,j,k-1) - g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(th0(i,j,k)+th0(i,j,k-1)))
        endif

        prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))
        qv0(i,j,k)=0.0
        qc0(i,j,k)=0.0
        thl0(i,j,k)=th0(i,j,k)

    enddo
    enddo
    enddo

    return
end subroutine mixeddrylayer_base

!-----------------------------------------------------------------------------
!
!  subroutine marginalcld_base:  Marginal Cloud Background w/ const N^2.
!  &param8 inputs :
!            var1 : Surface pressure [Pa]
!            var2 : Surface temperature [K]
!            var3 : Constant N^2 [1/s^2] 
!            var4 : 
!            var5 : 
!            var6 : 
!            var7 : 
!            var8 : 
!            var9 : 
!
!  NOTES: BLG Nov 2024 (Modified from RPW)
!         -Constant unsaturated N^2 ( d/dz log(thr) )
!         -Everywhere at saturation (r^* = r_T)
!
!-----------------------------------------------------------------------------

subroutine marginalcld_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants
    use cm1libs , only : rslf

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0
    real, dimension(ib:ie,jb:je,kb:ke) :: thv0, thl0, rst0

    real :: psfc, Tsfc, pisfc, thsfc, qvsfc, thvsfc
    real :: ns, Tlast

    integer :: i,j,k,n

    print *, 'Generating Marginal Cloud Base State'

    ! Define Surface conditions
    psfc = var1
    Tsfc = var2

    ! Define Brunt-Vaisala frequency N^2
    ns =  var3
    
    ! Calculate other surface values
    pisfc = (psfc/p00)**(rd/cp)
    thsfc = Tsfc/pisfc
    qvsfc = rslf(psfc, tsfc)
    thvsfc = thsfc*(1.0 + reps*qvsfc)/(1.0 + qvsfc)
    
    if(dowr) write(outfile,*)
    do j=jb,je
        do i=ib,ie
        do k=1,ke
            ! DEFINE \theta_v here for constant Nm^2 (unsaturated)
            thv0(i,j,k) = thvsfc*exp(ns/g*zh(i,j,k));

            ! Discretely step pressure forward
            if (k.eq.1) then 
                pi0(i,j,k)= pisfc - g*zh(i,j,k)/(cp*0.5*(thv0(i,j,k)+thvsfc))
            else 
                pi0(i,j,k)= pi0(i,j,k-1) - g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(thv0(i,j,k)+thv0(i,j,k-1)))
            endif
            
            prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))

        !Nonlinear root finding for theta_l and r*
        Tlast = 0
        n=1
        !Initial guess at T
        T0(i,j,k) = thv0(i,j,k)*pi0(i,j,k)

        do while((abs(Tlast-T0(i,j,k))/T0(i,j,k)).ge.1e-6)
            n=n+1
            !Check for infinite loop
            if (n.gt.100.0) stop 1111
            Tlast = T0(i,j,k)
            !!! calc new r* and T from thr
            qv0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))
            ! update T
            T0(i,j,k) = thv0(i,j,k)*pi0(i,j,k)/(1.0 + reps*qv0(i,j,k))*(1.0 + qv0(i,j,k))
        enddo

        !dry theta for CM1
        th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)

        !background thl
        thl0(i,j,k) = T0(i,j,k)*(p00/prs0(i,j,k))**((rd+qv0(i,j,k)*rv)/(cp+qv0(i,j,k)*cpv))              

        enddo  ! enddo for k loop
        enddo  ! enddo for i loop
    enddo  ! enddo for j loop

    return
end subroutine marginalcld_base

!-----------------------------------------------------------------------------
!
!  subroutine cldoverclear_base:  Specify base-state arrays for cloud over
!                                 clear from Walsh and Muraki (2024).
!  &param8 inputs :
!            var1 : Surface Pressure [Pa]
!            var2 : Surface Temperature [K]
!            var3 : Constant N^2 (unsaturated) [1/s^2]
!            var4 : Height of cloud edge [m]
!            var5 : 
!            var6 : 
!            var7 : liquid water slope at cloud edge [1/m]
!            var8 : 
!            var9 :
!
!  NOTES: BLG 07 June 2025
!
!-----------------------------------------------------------------------------

subroutine cldoverclear_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants
    use cm1libs , only : rslf

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0
    real, dimension(ib:ie,jb:je,kb:ke) :: thr0, thl0, rst0, qt0
    
    integer :: i,j,k,n, kbot, ktop

    real :: psfc, Tsfc, pisfc, thsfc, qvsfc, thrsfc
    real :: ns, Hc, rhslope, Tlast, lambc, lv_T
    real :: picld, thrcld, rstcld

    print *, 'Generating Base State for cloud over clear - Walsh and Muraki (2024)'

    ! Save user input variables (set in &param8 in namelist.input)
    psfc = var1
    tsfc = var2

    ns = var3
    Hc = var4
    rhslope = var7

    ! Calculate other surface values
    pisfc = (psfc/p00)**(rd/cp)
    thsfc = Tsfc/pisfc
    qvsfc = rslf(psfc, tsfc)
    thrsfc = thsfc*(1.0 + reps*qvsfc)/(1.0 + qvsfc)

    !!STAGE 1 - make * values!!
    do k=1,ke
    
        !!!!theta_v and p are exact from specified (approx) N^2 unsaturated
        !Bottom
        thr0(1,1,k) = thrsfc*exp(ns/g*zh(ib,jb,k))
        pi0(1,1,k) = pisfc - g**2/cp/thrsfc/ns*(1 - exp(-ns/g*zh(ib,jb,k)))
        prs0(1,1,k) = p00*(pi0(1,1,k)**(cp/rd))

        !!!!Nonlinear root finding for r* and T*
        Tlast = 0
        !Initial guess at T
        T0(1,1,k) = thr0(1,1,k)*pi0(1,1,k)

        do while((abs(Tlast-T0(1,1,k))/T0(1,1,k)).ge.1e-6)
            Tlast = T0(1,1,k)
            !!! calc new r* and T from thr
            rst0(1,1,k) = rslf(prs0(1,1,k),T0(1,1,k))
            ! update T
            T0(1,1,k) = thr0(1,1,k)*pi0(1,1,k)/(1.0 + reps*rst0(1,1,k))*(1.0 + rst0(1,1,k))
        enddo

        !initialize total thermodynamic fields
        do i=ib,ie
        do j=jb,je

            pi0(i,j,k)  = pi0(1,1,k)
            prs0(i,j,k) = prs0(1,1,k)
            T0(i,j,k)   = T0(1,1,k)
            thr0(i,j,k) = thr0(1,1,k)
            rst0(i,j,k) = rst0(1,1,k)

            !dry theta for CM1
            th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)
            !background thl
            thl0(i,j,k) = T0(i,j,k)*(p00/prs0(i,j,k))**((rd+rst0(i,j,k)*rv)/(cp+rst0(i,j,k)*cpv))
            
        enddo
        enddo

    enddo

    !!STAGE 2 - add water, keep thl the same, update pressure!!

    !first find cloud edge!
    k = 1
    do while(zh(ib,jb,k).lt.Hc)
        k = k+1
    enddo

    !Cld edge is a grid point
    IF(zh(ib,jb,k).eq.Hc)THEN
        kbot=k-1
        ktop=k+1
        picld=pi0(ib,jb,k)
        thrcld=thr0(ib,jb,k)
        rstcld=rst0(ib,jb,k)
    !Cld edge is between grid points
    ELSE
        kbot=k-1
        ktop=k
        !linear interp.
        picld=(pi0(ib,jb,kbot)*(Hc-zh(ib,jb,ktop)) + pi0(ib,jb,ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
        thrcld=(thr0(ib,jb,kbot)*(Hc-zh(ib,jb,ktop)) + thr0(ib,jb,ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
        rstcld=(rst0(ib,jb,kbot)*(Hc-zh(ib,jb,ktop)) + rst0(ib,jb,ktop)*(zh(ib,jb,kbot)-Hc))&
                /(zh(ib,jb,kbot)-zh(ib,jb,ktop))
    ENDIF


    do j=jb,je
    do i=ib,ie

    !start at cloud edge and go down (clear)
    do k=ktop,1,-1
        !New water
        qv0(i,j,k) = rst0(i,j,k)*(1 + rhslope*(zh(i,j,k) - Hc))
        qc0(i,j,k) = 0.0
        
        Tlast = 0.0
        
        !Nonlinear solve for pressure
        !T and th are exact with known p and thl in clear
        do while(abs((Tlast-th0(i,j,k))/th0(i,j,k)).gt.1e-6)
            Tlast = th0(i,j,k)
            th0(i,j,k) = thl0(i,j,k)*(p00/prs0(i,j,k))**&
                        ((rd*cpv-rv*cp)/(cp*(cp+qv0(i,j,k)*cpv))*qv0(i,j,k))
            thr0(i,j,k) = th0(i,j,k)*(1.0+reps*qv0(i,j,k))/(1.0+qv0(i,j,k))
            IF(k.eq.ktop)THEN
                pi0(i,j,k) = picld-g*(zh(i,j,k)-Hc)/(cp*0.5*(thr0(i,j,k)+thrcld))
            ELSE
                pi0(i,j,k) = pi0(i,j,k+1)-g*(zh(i,j,k)-zh(i,j,k+1))/(cp*0.5*(thr0(i,j,k)+thr0(i,j,k+1)))
            ENDIF
            prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))
        enddo
        T0(i,j,k)=th0(i,j,k)*pi0(i,j,k)
        
    enddo

    !start at cloud edge and go up (cloud)
    !For sufficiently hot temperature fixed point method of RPW fails
    !Use an approximate newton's method
    do k=ktop,ke
        qt0(i,j,k) = rst0(i,j,k)*(1 + rhslope*(zh(i,j,k) - Hc))

        !First guess using the liq. water function and linear T eq.
        lv_T = lv1 - lv2*T0(i,j,k)
        lambc = lv_T/(cp+qt0(i,j,k)*cpv)/T0(i,j,k)

        qc0(i,j,k) = (qt0(i,j,k) - rst0(i,j,k))*dlT_dlthl(prs0(i,j,k),T0(i,j,k))
        qv0(i,j,k) = qt0(i,j,k) - qc0(i,j,k)

        T0(i,j,k) = T0(i,j,k)*(1 + lambc*qc0(i,j,k))
        Tlast = 0.0

        !Nonlinear solve for pressure
        !Use Newton step to get T at each iteration
        do while(abs((Tlast-T0(i,j,k))/T0(i,j,k)).gt.3e-7)
            Tlast = T0(i,j,k)

            !Newton step
            T0(i,j,k) = T0(i,j,k)*(1 - log(thl_func(prs0(i,j,k),T0(i,j,k),qt0(i,j,k))/thl0(i,j,k))&
                                        *dlT_dlthl(prs0(i,j,k),T0(i,j,k)) )

            qv0(i,j,k) = rslf(prs0(i,j,k),T0(i,j,k))
            qc0(i,j,k) = qt0(i,j,k) - qv0(i,j,k)

            th0(i,j,k) = T0(i,j,k)/pi0(i,j,k)
            thr0(i,j,k) = th0(i,j,k)*(1.0+reps*qv0(i,j,k))/(1.0+qt0(i,j,k))

            IF(k.eq.ktop)THEN
                pi0(i,j,k) = picld-g*(zh(i,j,k)-Hc)/(cp*0.5*(thr0(i,j,k)+thrcld))
            ELSE
                pi0(i,j,k) = pi0(i,j,k-1)-g*(zh(i,j,k)-zh(i,j,k-1))/(cp*0.5*(thr0(i,j,k)+thr0(i,j,k-1)))
            ENDIF

            prs0(i,j,k)=p00*(pi0(i,j,k)**(cp/rd))

        enddo

    enddo

    enddo
    enddo

    return

    CONTAINS

    real function dlT_dlthl(prs,temp)
        real :: prs, temp, lamb, lambR, lambc, rtot, cptot, rst, lv_T
        
        lv_T  = lv1 - lv2*temp
        rst   = rslf(prs, temp)
        rtot  = rd + rst*rv
        cptot = cp + rst*cpv
        lambR = lv_T/rv/temp
        lambc = lv_T/cptot/temp
        lamb  = lambc*lambR*rtot/rd*rst

        dlT_dlthl = 1/(1+lamb)
    
    end function dlT_dlthl

    real function thl_func(prs,temp,qt)
        real :: prs, temp, qt, lambc, rtot, cptot, rst, lv_T, qv, qc

        lv_T  = lv1 - lv2*temp

        rst   = rslf(prs, temp)
        IF(rst.lt.qt)THEN
            qv = rst
            qc = qt-rst
        ELSE
            qv = qt
            qc = 0
        ENDIF

        rtot  = rd + qt*rv
        cptot = cp + qt*cpv
        lambc = lv_T/cptot/temp

        thl_func = temp*(p00/prs)**(rtot/cptot)*exp(-lambc*qc)&
                  *((rd+qv*rv)/rtot)**(rtot/cptot)*(qt/qv)**(qt*rv/cptot)

    end function thl_func

end subroutine cldoverclear_base

!-----------------------------------------------------------------------------
!
!  subroutine template_base:  Template for defining new base state initializations.
!
!  &param8 inputs :
!            var1 : Surface Pressure [Pa]
!            var2 : Surface Temperature [K]
!            var3 : 
!            var4 : 
!            var5 : 
!            var6 : 
!            var7 : 
!            var8 : 
!            var9 :
!
!  NOTES: BLG 06 Jun 2025
!
!-----------------------------------------------------------------------------

subroutine template_base(zh,pi0,prs0,th0,thl0,qv0,qc0,rst0,t0)

    use input
    use constants
    use cm1libs , only : rslf

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie,jb:je,kb:ke) :: pi0,prs0,th0,qv0,qc0,t0,thl0,rst0

    integer :: i,j,k,n

    real :: psfc, Tsfc

    ! Define Variables here

    print *, 'Generating Base State for ...'

    psfc = var1
    Tsfc = var2

    do j=jb,je
    do i=ib,ie
    do k=1,ke
        !loop through grid levels and set background state
    enddo
    enddo
    enddo

    return
end subroutine template_base


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Start definition of base state wind (iwnd option)  ccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


!-----------------------------------------------------------------------------
!
!  subroutine linearshear_wind:  
!
!  &param8 inputs :
!            var10 : Constant wind speed outside shear layer [m/s]
!            var11 : Height at centre of shear layer [m]
!            var12 : Half depth of shear layer [m]
!            var13 :
!
!  NOTES: BLG 07 Jun 2025
!         -Linear shear layer from Hc-Hd -> Hc+Hd
!         -Symmetric shear with mean 0
!
!-----------------------------------------------------------------------------

subroutine linearshear_wind(zh,u0,v0)

    use input
    use constants

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie+1,jb:je,kb:ke) :: u0
    real, dimension(ib:ie,jb:je+1,kb:ke) :: v0

    integer :: i,j,k

    real :: uinf, Hc, Hd, zu, zv

    print *, 'Generating Background Wind for Linear Shear Layer'

        uinf = var10
        Hc   = var11
        Hd   = var12

        do k=1,nk
        do j=0,nj+1
        do i=0,ni+2

            zu=0.5*(zh(i-1,j,k)+zh(i,j,k))
            if(zu.le.(Hc-Hd))then
                u0(i,j,k) = -uinf
            elseif(zu.ge.(Hc+Hd))then
                u0(i,j,k) = uinf
            else
                u0(i,j,k)=uinf*(zu-Hc)/Hd
            endif

        enddo
        enddo
        enddo

        do k=1,nk
        do j=0,nj+2
        do i=0,ni+1

            zv=0.5*(zh(i,j-1,k)+zh(i,j,k))
            v0(i,j,k)=0.0

        enddo
        enddo
        enddo

end subroutine linearshear_wind

!-----------------------------------------------------------------------------
!
!  subroutine template_wind: Template for defining base state winds.
!
!  &param8 inputs :
!           var10 : 
!           var11 : 
!           var12 : 
!           var13 :
!
!  NOTES: BLG 07 Jun 2025
!
!-----------------------------------------------------------------------------

subroutine template_wind(zh,u0,v0)

    use input
    use constants

    implicit none

    real, dimension(ib:ie,jb:je,kb:ke) :: zh
    real, dimension(ib:ie+1,jb:je,kb:ke) :: u0
    real, dimension(ib:ie,jb:je+1,kb:ke) :: v0

    integer :: i,j,k
    
    real :: zu, zv

    print *, 'Generating Background Wind for...'


        do k=1,nk
        do j=0,nj+1
        do i=0,ni+2

            !Set background wind in x-direction
            zu=0.5*(zh(i-1,j,k)+zh(i,j,k))
            u0(i,j,k)=0.0

        enddo
        enddo
        enddo

        do k=1,nk
        do j=0,nj+2
        do i=0,ni+1

            !Set background wind in y-direction
            zv=0.5*(zh(i,j-1,k)+zh(i,j,k))
            v0(i,j,k)=0.0

        enddo
        enddo
        enddo

end subroutine template_wind

END MODULE base_blg_module
    