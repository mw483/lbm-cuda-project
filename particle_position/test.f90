program particle_position
   implicit none
   integer :: i,j,id
   real,parameter :: ui=0.0,vi=0.0,wi=0.13
   real,parameter :: xref=240.0,yref=288.0,zref=1.0,a=0.1,basement=0.0
   integer,parameter :: imax=3,jmax=50
   id = 1
   do i = 1,imax
     do j = 1,jmax
       id = id + 3000
       write(*,700)xref,yref,zref*a**i+basement,ui,vi,wi,1,id
     end do
   end do
700 format(f10.6,x,f10.6,x,f10.6,x,i1,x,f10.6,x,f10.6,x,i1,x,i12)
end program

