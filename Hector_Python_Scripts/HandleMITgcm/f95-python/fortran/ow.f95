subroutine okubo(x,s,dataout,m,n)
use omp_lib
!f2py threadssafe
implicit none
!----------------------
! vorticity
!---------------------

integer :: m
integer :: n
real(8), intent(in), dimension(m,n) :: x
real(8), intent(in), dimension(m,n) :: s
real(8), intent(out), dimension(m,n) :: dataout
integer :: j
integer :: k
!! f2py declarations
!f2py real(8), intent(in), dimension(m,n) :: x
!f2py real(8), intent(in), dimension(m,n) :: s
!f2py integer, intent(in) :: m
!f2py integer, intent(in) :: n
!f2py real(8), intent(out), dimension(m,n) :: dataout

      do k = 1,n
        do j = 1,m
         dataout(j,k) = 0.0
        end do
      end do

!$OMP PARALLEL DO  private ( J, K) NUM_THREADS(2)
      do j = 1,m
        do k = 1,n
          dataout(j,k)=s(j,k) - x(j,k)
        end do
      end do
!$OMP END PARALLEL DO

END subroutine okubo


