subroutine s(s1,s2,dataout,m,n)
use omp_lib
!f2py threadssafe
implicit none
!----------------------
! vorticity
!---------------------

integer :: m
integer :: n
real(8), intent(in), dimension(m,n) :: s1
real(8), intent(in), dimension(m,n) :: s2
real(8), intent(out), dimension(m,n) :: dataout
integer :: j
integer :: k
!! f2py declarations
!f2py real(8), intent(in), dimension(m,n) :: s1
!f2py real(8), intent(in), dimension(m,n) :: s2
!f2py integer, intent(in) :: m
!f2py integer, intent(in) :: n
!f2py real(8), intent(out), dimension(m,n) :: dataout

      do k = 1,n
        do j = 1,m
         dataout(j,k) = 0.0
        end do
      end do

!$OMP PARALLEL DO  private ( J, K) NUM_THREADS(2)
      do j = 2,m-1
        do k = 2,n-1
          dataout(j,k)=s2(j,k)+0.25*(s1(j+1,k+1)+s1(j+1,k)+s1(j,k)+s1(j,k+1))
        end do
      end do
!$OMP END PARALLEL DO

END subroutine s
