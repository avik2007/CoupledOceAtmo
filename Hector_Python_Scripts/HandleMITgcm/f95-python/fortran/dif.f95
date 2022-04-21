subroutine d(array,dx,dataout,m,n)
  use omp_lib
  implicit none


  integer, intent(in) :: m
  integer, intent(in) :: n
  real(8) :: array(m,n)
  real(8) :: dx(m,n)
  real(8) :: dataout(m,n)
  !!f2py declarations
!f2py integer, intent(in) :: m,n
!f2py real(8), intent(in), dimension(m,n) :: array
!f2py real(8), intent(in), dimensin(m,n) :: dx
!f2py real(8), intent(out), dimension(m,n) :: dataout


  integer :: j
  integer :: k
  !integer :: i

  do j = 1,m
    dataout(j,1) = 0.0
    dataout(j,n) = 0.0
  end do



  do k = 1,n
    dataout(1,k) = 0.0
    dataout(m,k) = 0.0
  end do


!$OMP PARALLEL SHARED ( dataout ) PRIVATE ( I, J )
!$OMP DO

  do j = 1,m-1
    do k = 1,n-1
       dataout(j,k) = (array(j+1,k) - array(j,k)) - &
       (array(j,k+1) - array(j,k))/(dx(j,k))
    end do
  end do
!$OMP END DO
!$ OMP END PARALLEL
end subroutine d


