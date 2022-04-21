subroutine st(u,v,dataout,dx,dy,raz,m,n)
use omp_lib
!f2py threadssafe
implicit none
!----------------------
! vorticity
!---------------------

integer :: m
integer :: n
real(8), intent(in), dimension(m,n) :: u
real(8), intent(in), dimension(m,n) :: v
real(8), intent(in), dimension(m,n) :: dx
real(8), intent(in), dimension(m,n) :: dy
real(8), intent(out), dimension(m,n) :: dataout
real(8), intent(in), dimension(m,n) :: raz
integer :: j
integer :: k
!! f2py declarations
!f2py real(8), intent(in), dimension(m,n) :: u
!f2py real(8), intent(in), dimension(m,n) :: v
!f2py real(8), intent(in), dimension(m,n) :: dx
!f2py real(8), intent(in), dimension(m,n) :: dy
!f2py real(8), intent(in), dimension(m,n) :: raz
!f2py integer, intent(in) :: m
!f2py integer, intent(in) :: n
!f2py real(8), intent(out), dimension(m,n) :: dataout

      do k = 1,n
         dataout(1,k) = 0.0
         dataout(m,k) = 0.0
      end do

      do j = 1,m
         dataout(j,1) = 0.0
         dataout(j,n) = 0.0
      end do

!$OMP PARALLEL DO  private ( J, K) NUM_THREADS(2)
      do j = 1,m-1
        do k = 1,n-1
          dataout(j,k)=(1/raz(j,k))*(dy(j,k+1)*u(j,k+1)-dy(j,k)*u(j,k)-dx(j+1,k)*v(j+1,k)+dx(j,k)*v(j,k))
        end do
      end do
!$OMP END PARALLEL DO

END subroutine st

