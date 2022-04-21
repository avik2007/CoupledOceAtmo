subroutine del2(array,rms,n)
implicit none

integer, intent(in) :: n
double precision, intent(in), dimension(n) :: array
double precision, intent(out), dimension(1) :: rms

rms = sqrt(sum(array**2)/n)

end subroutine  del2
