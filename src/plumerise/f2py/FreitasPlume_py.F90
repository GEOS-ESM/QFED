! **********************************************
!
!  Simple f77 wrapper for the Python interface.
!
! **********************************************

 
subroutine plume ( km, u, v, T, q, delp, ptop, hflux_kW, area, & 
                   p, z, k, rc)

   use FreitasPlume_Mod

   implicit none

!  !ARGUMENTS:

   integer, intent(in)  :: km             ! number of vertical layers

   real,    intent(in)  :: u(km)          ! zonal wind (m/s)
   real,    intent(in)  :: v(km)          ! meridional wind (m/s)
   real,    intent(in)  :: T(km)          ! potential temperature [K]
   real,    intent(in)  :: q(km)          ! specific humidity [kg/kg]
   real,    intent(in)  :: delp(km)       ! pressure thickness [Pa]
   real,    intent(in)  :: ptop           ! top edge pressure [Pa]
   real,    intent(in)  :: area           ! fire area [m^2]
   real,    intent(in)  :: hflux_kW       ! fire heat flux [kW/m2]
   
   real,    intent(out) :: p              ! upper plume pressure
   real,    intent(out) :: z              ! upper plume height
   integer, intent(out) :: k              ! upper plume vertical index
   integer, intent(out) :: rc             ! error code

 !                       ----

!  Local variables
!  ---------------
   type(FreitasPlume) :: pr
   
!                                ----

!  Initialize
!  ----------
   call FreitasPlume_Initialize(pr)

!  Run
!  ---
   call FreitasPlume_Run (pr, km,     &
                       u, v, T, q, delp, ptop, &
                       area, hflux_kW,   &
                       z, p, k )

!  Finalize
!  ---------
   call FreitasPlume_Finalize(pr)

   rc = 0

 end subroutine plume

subroutine PlumesVMD ( km, nf, u, v, T, q, delp, ptop, hflux_kW, area, & 
                       z_i, z_d, z_a, z_f, z_plume, w_plume, rc)

   use FreitasPlume_Mod
   use omp_lib
   
   implicit none  

!  !ARGUMENTS:

   integer, parameter :: nkp_ = 200          ! This must be explicit for f2py
   
   integer, intent(in)  :: km                ! number of vertical layers
   integer, intent(in)  :: nf                ! number of fires

   real,    intent(in)  :: u(km,nf)          ! zonal wind (m/s)
   real,    intent(in)  :: v(km,nf)          ! meridional wind (m/s)
   real,    intent(in)  :: T(km,nf)          ! potential temperature [K]
   real,    intent(in)  :: q(km,nf)          ! specific humidity [kg/kg]
   real,    intent(in)  :: delp(km,nf)       ! pressure thickness [Pa]
   real,    intent(in)  :: ptop              ! top edge pressure [Pa]
   real,    intent(in)  :: area(nf)          ! fire area [m^2]
   real,    intent(in)  :: hflux_kW(nf)      ! fire heat flux [kW/m2]
   
   real,    intent(out) :: z_i(nf)  ! height of maximum W (bottom of plume)
   real,    intent(out) :: z_d(nf)  ! height of maximum detrainment
   real,    intent(out) :: z_a(nf)  ! average height in (z_i,z_f), weighted by -dw/dz
   real,    intent(out) :: z_f(nf)  ! height where w<1 (top of plume)

   real,    intent(out) :: z_plume(nkp_)    ! native vertical levels (same for all fires)
   real,    intent(out) :: w_plume(nkp_,nf) ! native vertical velocity
   
   
   integer, intent(out) :: rc(nf)

 !                       ----

   integer :: n

   if ( nkp_ .ne. nkp ) then
      print *, 'PlumesVMD: internal error, fix nkp', nkp, nkp_
      rc = 1
      return
   endif

   !ams print *, 'PlumesVMD: Open MP maximum number of threads: ', OMP_get_max_threads()

!$OMP PARALLEL DO   &
!$OMP    PRIVATE(n) &
!$OMP    SHARED(km,nf,ptop,u,v,T,q,delp,hflux_kW,area,z_i,z_d,z_a,z_f,rc)
   
   do n = 1, nf

      !ams print *, n
      
      call plumeVMD ( km,                                              &
                      u(:,n), v(:,n), T(:,n), q(:,n), delp(:,n), ptop, &
                      hflux_kW(n), area(n),                            & 
                      z_i(n), z_d(n), z_a(n), z_f(n),                  &
                      z_plume, w_plume(:,n), rc(n) )

   end do

!$OMP END PARALLEL DO
                 
end subroutine PlumesVMD


!---
subroutine plumeVMD ( km, u, v, T, q, delp, ptop, hflux_kW, area, & 
                      z_i, z_d, z_a, z_f, z_plume, w_plume, rc)

   use FreitasPlume_Mod
  
   implicit none

!  !ARGUMENTS:

   integer, parameter :: nkp_ = 200       ! This must be explicit for f2py

   integer, intent(in)  :: km             ! number of vertical layers

   real,    intent(in)  :: u(km)          ! zonal wind (m/s)
   real,    intent(in)  :: v(km)          ! meridional wind (m/s)
   real,    intent(in)  :: T(km)          ! potential temperature [K]
   real,    intent(in)  :: q(km)          ! specific humidity [kg/kg]
   real,    intent(in)  :: delp(km)       ! pressure thickness [Pa]
   real,    intent(in)  :: ptop           ! top edge pressure [Pa]
   real,    intent(in)  :: area           ! fire area [m^2]
   real,    intent(in)  :: hflux_kW       ! fire heat flux [kW/m2]
   
   real,    intent(out) :: z_i  ! height of maximum W (bottom of plume)
   real,    intent(out) :: z_d  ! height of maximum detrainment
   real,    intent(out) :: z_a  ! average height in (z_i,z_f), weighted by -dw/dz
   real,    intent(out) :: z_f  ! height where w<1 (top of plume)

   real,    intent(out) :: z_plume(nkp_) ! native vertical levels
   real,    intent(out) :: w_plume(nkp_) ! native vertical velocity
   
   integer, intent(out) :: rc

 !                       ----


!  Local variables
!  ---------------
   type(FreitasPlume) :: pr
   
!                                ----

   if ( nkp_ .ne. nkp ) then
      print *, 'PlumeVMD: internal error, fix nkp', nkp, nkp_
      rc = 1
      return
   endif
        
   
!  Initialize
!  ----------
   call FreitasPlume_Initialize(pr)

!  Run
!  ---
   call FreitasPlume_Run (pr, km,                 &
                          u, v, T, q, delp, ptop, &
                          area, hflux_kW,         &
                          z_i, z_d, z_a, z_f,     &
                          z_plume, w_plume )

!
!  Find bottom, mid and top plume height. If using the parabolic VMD as in getVMD() below,
!  there are several options:
!
!  a) Like Saulo:
!                     z_c   = (z_f+z_i)/2
!                     delta = (z_f-z_i)/2
!
!  b) Preserve bottom half:
!                     z_c   = z_d
!                     delta = z_d - z_i
!
!  c) Preserve upper half:
!                     z_c   = z_d
!                     delta = z_f - z_d
!
!                       ---


!  Finalize
!  ---------
   call FreitasPlume_Finalize(pr)

   rc = 0

 end subroutine plumeVMD
 

!..................................................................................

subroutine getVMD(km,nf,z,z_c,delta,v) 

!
! Computes normalized vertical mass distribution (VMD) given z_c and delta. Assumes a gaussian.
!
   implicit NONE
   integer, intent(in)  :: km        ! number of vertical layers
   integer, intent(in)  :: nf        ! number of fires
   real,    intent(in)  :: z(km,nf)  ! height above surface [m]
   real,    intent(in)  :: z_c(nf)   ! level of maximum detrainment (center of plume)
   real,    intent(in)  :: delta(nf) ! width of vertical mass distribution
   
   real,    intent(out) :: v(km,nf)  ! normalized vertical mass distribution
!                   ---

  integer :: n
  real*8  :: v_(km), z_(km), dz_(km)
  real*8  :: vnorm

  v = 0.0

!$OMP PARALLEL DO         &
!$OMP    PRIVATE(n,z_,v_,vnorm) &
!$OMP    SHARED(v,z,z_c,delta,nf)

!  Analytic function
!  -----------------
   do n = 1, nf
   
    v_ = 0.0
    z_ = z(:,n)
    dz_ = abs(z_ - z_c(n))
    where ( dz_ <= delta(n) )
            v_ = 3.*(delta(n)**2 - dz_**2)/(4.*delta(n)**3) 
    end where

!   Ensure normalization on grid
!   ----------------------------
    vnorm = sum(v_)
    if ( vnorm > 0.0 ) then
         v(:,n) = v_ / vnorm
    end if

   end do

!$OMP END PARALLEL DO

!ams  print *, 'km, nf = ', km, nf
!ams  print *, '     z = ', minval(z), maxval(z)
!ams  print *, '   z_c = ', minval(z_c), maxval(z_c)
!ams  print *, ' delta = ', minval(delta), maxval(delta)
!ams  print *, '     v = ', minval(v), maxval(v)
  
 end subroutine getVMD

!..................................................................................

subroutine plumeBiome ( km, u, v, T, q, delp, ptop, area, ibiome, & 
                        p1, p2, z1, z2, k1, k2, rc)

   use FreitasPlume_Mod

   implicit none

!  !ARGUMENTS:

   integer, intent(in)  :: km             ! number of vertical layers

   real,    intent(in)  :: u(km)          ! zonal wind (m/s)
   real,    intent(in)  :: v(km)          ! meridional wind (m/s)
   real,    intent(in)  :: T(km)          ! potential temperature [K]
   real,    intent(in)  :: q(km)          ! specific humidity [kg/kg]
   real,    intent(in)  :: delp(km)       ! pressure thickness [Pa]
   real,    intent(in)  :: ptop           ! top edge pressure [Pa]
   real,    intent(in)  :: area           ! fire area [m^2]
   integer, intent(in)  :: ibiome         ! biome index
   
   real,    intent(out) :: p1             ! lower pressure
   real,    intent(out) :: p2             ! upper pressure
   real,    intent(out) :: z1             ! lower plume height
   real,    intent(out) :: z2             ! upper plume height
   integer, intent(out) :: k1             ! upper plume vertical index
   integer, intent(out) :: k2             ! lower plume vertical index
   integer, intent(out) :: rc             ! error code

!                       ----

!  Local variables
!  ---------------
   type(FreitasPlume) :: pr
   
   real    :: p1_p(N_BIOME),p2_p(N_BIOME) 
   real    :: z1_p(N_BIOME),z2_p(N_BIOME) 
   integer :: k1s(N_BIOME), k2s(N_BIOME)
   real    :: areas(N_BIOME)

!                                ----

!  Biome check
!  -----------
   if ( ibiome < 1 .OR. ibiome > N_BIOME ) then
      rc = 1
      return
   else
      areas = 0.0
      areas(ibiome) = area
   end if

!   print *, '----'
!   print *, 'f2py: km, ibiome, ptop, areas = ', km, ibiome, ptop, area 
!   print *, 'f2py:      T min/max   = ', minval(T), maxval(T)
!   print *, 'f2py:      q min/max   = ', minval(q), maxval(q)
!   print *, 'f2py:   delp min/max   = ', minval(delp), maxval(delp)

!  Initialize
!  ----------
   call FreitasPlume_Initialize(pr)

!  Run
!  ---
   call FreitasPlume_Run (pr, km, u, v, T, q, delp, ptop, areas, &
                       z1_plume=z1_p, z2_plume=z2_p,    &
                       p1_plume=p1_p, p2_plume=p2_p,    &
                       k1_plume=k1s,  k2_plume=k2s)
                           
!  Finalize
!  ---------
   call FreitasPlume_Finalize(pr)

   p1 = p1_p(ibiome)  ! bottom pressure
   p2 = p2_p(ibiome)  ! top pressure
   z1 = z1_p(ibiome)
   z2 = z2_p(ibiome)
   k1 = k1s(ibiome) 
   k2 = k2s(ibiome)
   
   rc = 0

!  write(*,'(a,2F10.2,2I4,2F10.2)') 'f2py: p, k, z = ', p1, p2, k1, k2, z1, z2 

 end subroutine plumeBiome

 !..................................................................................

  subroutine kPBL( k_pbl, km, T, q, delp, ptop, pblh )

   use rconstants
   implicit NONE

   integer, intent(in) :: km                 ! No. vertical layers
   real,    intent(in) :: T(km)              ! Dry temperature [K]
   real,    intent(in) :: q(km)              ! Specific humidity [kg/kg]
   real,    intent(in) :: delp(km)           ! layer pressure thickness [Pa]
   real,    intent(in) :: ptop               ! top (edge) pressure [Pa]
   real,    intent(in) :: pblh               ! PBL height [m]
   
   integer, intent(out) :: k_pbl             ! vertical layer index of PBL

!
! Returns the vertical layer index of the PBL.
!

!
! ------------------------------- pe(k),   he(k)
!
! ............................... pm(k),   hm(k), delp(k)
!
! ------------------------------- pe(k+1), he(k+1)
!

   integer :: k
   real :: Tv, kappa = rocp
   real :: delh(km), pe(km+1), he(km+1), mixr(km)

!  Mixing ratio from specific humidity
!  -----------------------------------
   mixr = q / ( 1.0 - q )

!  Construct edge pressures
!  ------------------------
   pe(1) = ptop
   do k = 1, km
      pe(k+1) = pe(k) + delp(k)
   end do

!  Construct mid-layer pressures and layer thickness
!  -------------------------------------------------
   do k = 1, km
      Tv = T(k) * ( 1 + 0.61 * mixr(k) )
      delh(k)  = Rgas * Tv * log(pe(k+1)/pe(k)) / g
    end do

!  Compute Geo-potential height at edges
!  -------------------------------------
   he(km+1) = 0.0  ! set the surface to zero height
   do k = km, 1, -1
      he(k) = he(k+1) + delh(k)
   end do

!  Now find the PBL layer
!  ----------------------
   k_pbl = -1  ! so that we can check later for abnormal behavior 
   do k = km, 1, -1
      if ( he(k) >= pblh ) then
           k_pbl = k
           exit
      end if
   end do

 end subroutine kPBL

 !----------------------------------------------------------------------------

subroutine setNumThreads ( num_threads )
  use omp_lib
  integer(kind = OMP_integer_kind), intent(in) :: num_threads 
  call OMP_set_num_threads(num_threads)
  print *, 'Open MP maximum number of threads: ', OMP_get_max_threads()
end subroutine setNumThreads

 subroutine ompTest(N,num_threads,result)

  use omp_lib
  
  integer, intent(in) :: N             ! problem size
  integer(kind = OMP_integer_kind), intent(in) :: num_threads 

  real, intent(out)   :: result(N)
  
  !   ---

  real :: x, dx


   call OMP_set_num_threads(num_threads)
   print *, '*** Num threads: ', OMP_get_num_threads()
   print *, '*** Max threads: ', OMP_get_max_threads()
  
  dx = 3.141515 / (N-1)
  if (num_threads>1) then

!$OMP PARALLEL DO           &
!$OMP    PRIVATE(i,x)       &
!$OMP    SHARED(n,result)
   do i = 1, N
      x = (i-1) * dx
      result(i) = sin(x) * cos(x) / (sin(x)**2 + cos(x)**2)
   end do

!$OMP END PARALLEL DO
     
else

   do i = 1, N
      x = (i-1) * dx
      result(i) = sin(x) * cos(x) / (sin(x)**2 + cos(x)**2)
   end do   

  end if

end subroutine ompTest
