subroutine ke(u,v,kinetic,m,n)
use omp_lib
!f2py threadssafe
implicit none
!---------------------------
! Kinetic Energy
! in vector invariant form
!---------------------------
integer :: m,n
real(8), intent(in), dimension(m,n) :: u
real(8), intent(in), dimension(m,n) :: v
real(8), intent(out), dimension(m,n) :: kinetic
integer :: i,j
!! f2py declarations
!f2py real(8), intent(in), dimension(m,n) :: u
!f2py real(8), intent(in), dimension(m,n) :: v
!f2py integer, intent(in) :: m
!f2py integer, intent(in) :: n
!f2py real(8), intent(out), dimension(m,n) :: kinetic

   do i = 1,n
      do j = 1,m
        kinetic(j,i) = 0.0
      end do
   end do

!$OMP PARALLEL DO  private ( I, j) NUM_THREADS(2)
   do i = 2,n-1
     do j =2,m-1
       kinetic(j,i) = 0.25*((u(j,i)*u(j,i)+u(j,i+1)*u(j,i+1))+(v(j,i)*v(j,i)+v(j+1,i)*v(j+1,i)))
     end do
   end do
!$OMP END PARALLEL DO

end subroutine ke 
