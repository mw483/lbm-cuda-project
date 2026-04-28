program generate_cubes
   integer,parameter :: im=64,icmax=16,irmax=16,irep=100,imax=352,      &
                        jm=8,jcmax=16,jrmax=16,jrep=100,jmax=288
   real,parameter :: bh=32.0 ,basement=2.0
   real,dimension(1:20000,1:20000) :: h
   h = basement
   do ir = 0,irep-1
     do jr = 0,jrep-1
       do ic = 0,icmax-1
         do jc = 0,jcmax-1
           i = im + ir * (icmax + irmax) + (ic + 1)
           j = jm + jr * (jcmax + jrmax) + (jc + 1)
           h(i,j) = h(i,j) + bh
         end do
       end do
     end do
   end do

   open(11,file='_p3d')
   do j = 1,jmax
     do i = 1,imax
       write(11,700,advance='no')h(i,j)
     end do
     write(11,*)
   end do
   close(11)
700 format(f5.1)
end program generate_cubes

