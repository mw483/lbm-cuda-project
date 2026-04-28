#ifndef OUTPUTPARAVIEWINC_H_
#define OUTPUTPARAVIEWINC_H_


#include "outputParaview.h"


template <typename T>
void	Paraview::
combine_data (
	T		*f_out,
	int		lnx,
	int		lny,
	int		lnz,
	T		*f_in,
	int		nx,
	int		ny,
	int		nz
	)
{
	MPI_Status	stat;

	const int	nn = nx*ny*nz;

	const int	rank_x = rank_x_,
		  		rank_y = rank_y_,
		  		rank_z = rank_z_;

	const int	ncpu_x = ncpu_x_,
		  		ncpu_y = ncpu_y_;
//		  		ncpu_z = ncpu_z_;

	const int	ncpu_div_x = ncpu_div_x_,
				ncpu_div_y = ncpu_div_y_,
				ncpu_div_z = ncpu_div_z_;

	const int	lid_x = rank_x / ncpu_div_x,
		  		lid_y = rank_y / ncpu_div_y,
		  		lid_z = rank_z / ncpu_div_z;


	// file index
	const int	fid_x = rank_x % ncpu_div_x,
		  		fid_y = rank_y % ncpu_div_y,
		  		fid_z = rank_z % ncpu_div_z;
	const int	fid   = fid_x + ncpu_div_x*fid_y + ncpu_div_x*ncpu_div_y*fid_z;


	if (fid == 0) {
		T	*f_tmp = new T[nn];

		for (int rk=0; rk<ncpu_div_z; rk++) {
			for (int rj=0; rj<ncpu_div_y; rj++) {
				for (int ri=0; ri<ncpu_div_x; ri++) {
					const int	id_rank = (rank_x+ri) + ncpu_x*(rank_y+rj) + ncpu_x*ncpu_y*(rank_z+rk);

					if (ri == 0 && rj == 0 && rk == 0) {
						memcpy(f_tmp, f_in,   sizeof(T)*(nn));
					}
					else {
						MPI_Recv(f_tmp, (nn), MFLOAT, id_rank, 0, MPI_COMM_WORLD, &stat);
					}

					// copy
					for (int k=halo_; k<nz-halo_; k++) {
						for (int j=halo_; j<ny-halo_; j++) {
							for (int i=halo_; i<nx-halo_; i++) {
								const int	id  = i + nx*j + nx*ny*k;

								const int	pid_x = (i-halo_) + (nx - 2*halo_)*ri;
								const int	pid_y = (j-halo_) + (ny - 2*halo_)*rj;
								const int	pid_z = (k-halo_) + (nz - 2*halo_)*rk;

								const int	pid   = pid_x + lnx*pid_y + lnx*lny*pid_z;

								f_out[pid] = f_tmp[id];
							}
						}
					}
				}
			}
		}

		delete [] f_tmp;
	}
	else {
		const int	rank_fid0 = (lid_x*ncpu_div_x) + ncpu_x*(lid_y*ncpu_div_y) + ncpu_x*ncpu_y*(lid_z*ncpu_div_z);

		MPI_Send(f_in, (nn), MFLOAT,  rank_fid0, 0, MPI_COMM_WORLD);
	}
}


#endif
