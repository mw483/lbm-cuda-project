#include "Set_Obstacle.h"

#include "defineBoundaryFlag.h"


// obstacle
void Read_Obstacle_Data(MPIinfo *mpi, Domain *cdo, char *status)
{
	const int	nx = cdo->nx,
				ny = cdo->ny,
				nz = cdo->nz;
	const int	nxy = nx*ny;

	// mpi
	const int	rank_y = mpi->rank_y,
				rank_z = mpi->rank_z;

	const int	nyg = cdo->nyg,
				nzg = cdo->nzg;
	if (mpi->rank == 0)	{ std::cout << "(nxg, nyg, nzg) = ("  << cdo->nxg << ", " << nyg << ", " << nzg << ")\n"; }
	MPI_Barrier(MPI_COMM_WORLD);

	int		*hight_map_g;
	hight_map_g = new int[nyg*nzg];
	for (int i=0; i<nyg*nzg; i++) { hight_map_g[i] = 0; }

	// fin
	if (mpi->rank == 0) {
		// deco-boco *****
		const int	deco = 5;
		for (int k=0; k<nzg; k++) {
			for (int j=0; j<nyg; j++) {
				const int	id_map_g =  j     + nyg * k;

				hight_map_g[id_map_g] = ( abs(   (j+k)%(2*deco)-deco)%deco 
										+ abs(abs(j-k)%(2*deco)-deco)%deco)/2;

//				if ((j%5==0 && k%5==0) || (j%5==1 && k%5==0) || (j%5==0 && k%5==1) || (j%5==1 && k%5==1)) {
//					hight_map_g[id_map_g] = deco;
//				}
			}
		}
		// deco-boco *****


		// fin
		std::cout << "read mapdata : start\n";
		char name[100];

		std::ifstream fin;
		sprintf(name, "./mapdata/input.dat");

		fin.open(name, std::ios::in);
		if(!fin) { std::cout << "file is not opened\n"; }


		// resolution
		int		resolution_ny,
				resolution_nz; // obstacle size

		fin >> resolution_ny;
		fin >> resolution_nz;

		std::cout << "map size = " << resolution_ny << " x " << resolution_nz << "\n";

		// map data
		int		*hight_map;
		hight_map = new int[resolution_ny*resolution_nz];
		for (int i=0; i<resolution_ny*resolution_nz; i++)	{ hight_map[i] = 0; }
		for (int i=0; i<resolution_ny*resolution_nz; i++)	{ fin >> hight_map[i]; }
		fin.close();

		std::cout << "read mapdata : finish\n";

		// reverse *********************
		for (int k=0; k<resolution_nz; k++) {
			std::reverse( &hight_map[k*resolution_ny], &hight_map[k*resolution_ny] + resolution_ny );
		}
		// reverse *********************

		std::cout << "start : mapdata convert\n";
		// map -> map global
		int		map_offset_y = (nyg - resolution_ny)/2,
				map_offset_z = (nzg - resolution_nz)/2; // ( nyg < resolution ) -> offset < 0

		int		js = map_offset_y,
				je = js + resolution_ny;
		int		ks = map_offset_z,
				ke = ks + resolution_nz;

		for (int k=0; k<nzg; k++) {
			for (int j=0; j<nyg; j++) {
				if ( (j > js && j < je) && (k > ks && k < ke) ) {
					const int	id_map   = (j-js) + resolution_ny * (k-ks);
					const int	id_map_g =  j     + nyg           *  k;
	
					hight_map_g[id_map_g] = hight_map[id_map];
				}
			}
		}

		// fout map_offset
		std::ofstream fout;
		fout.open("./map_offset.dat");
		fout << map_offset_y << "\t" << map_offset_z << "\n";
		fout.close();

//		int		map_offset_y = 10,
//				map_offset_z = 10;
//		// map : center
//		if (resolution_ny < nyg)	{ map_offset_y = (nyg - resolution_ny)/2; }
//		if (resolution_nz < nzg)	{ map_offset_z = (nzg - resolution_nz)/2; }
//
//		// map > nx,ny
//		if (resolution_ny > nyg)	{ map_offset_y = 0; }
//		if (resolution_nz > nzg)	{ map_offset_z = 0; }
//
//		std::ofstream fout;
//		fout.open("./map_offset.dat");
//		fout << map_offset_y << "\t" << map_offset_z << "\n";
//		fout.close();
//
//		int		js = map_offset_y,
//				jf = js + resolution_ny;
//		int		ks = map_offset_z,
//				kf = ks + resolution_nz;
//
//		if (jf > nyg)	{ jf = nyg; }
//		if (kf > nzg)	{ kf = nzg; }
//
//		for (int k=ks; k<kf; k++) {
//			for (int j=js; j<jf; j++) {
//				const int	id_map   = (j-js) + resolution_ny * (k-ks);
//				const int	id_map_g =  j     + nyg           *  k;
//
//				hight_map_g[id_map_g] = hight_map[id_map];
//			}
//		}

		delete [] hight_map;
		std::cout << "end   : mapdata convert\n";
	} 
	MPI_Barrier(MPI_COMM_WORLD);

	// hight_map
	if (mpi->rank == 0) { std::cout << "start   : mapdata bcast\n"; }
	MPI_Bcast(hight_map_g, (nyg*nzg), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if (mpi->rank == 0) { std::cout << "end   : mapdata bcast\n"; }


	// map global > status
	const int	halo = 25;

	const int	Offset_Y = 1; // バグを誘発する記述 //
	const int	Offset_Z = 1;

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			const int	jg = ( rank_y*(ny-2*Offset_Y) + (j-Offset_Y) + nyg )%nyg;
			const int	kg = ( rank_z*(nz-2*Offset_Z) + (k-Offset_Z) + nzg )%nzg;

			const int	id_map = jg + kg * nyg;

			// map global
			int		hight = hight_map_g[id_map];

			// halo *****
			if (  (jg<halo || jg>nyg-halo)
			   || (kg<halo || kg>nzg-halo) ) {
				hight = 0;
			}
			// halo *****

			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nxy*k;

				// x
				const int	x = i;

				int		frg = STATUS_FLUID;
				if (i == 0    )				{ frg = STATUS_WALL; }
//				if (i == 0 || i == (nx-1))	{ frg = STATUS_WALL; }
				if (x <= hight)				{ frg = STATUS_WALL; }
				if (i == nx-1)				{ frg = STATUS_SLIP_X; }

				status[id] = frg;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


	delete [] hight_map_g;
}

// with levelset
void Set_Obstacle_Channel(MPIinfo *mpi, Domain *cdo, char *status, FLOAT *l_obs)
{
	const int	nx = cdo->nx,
				ny = cdo->ny,
				nz = cdo->nz;

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				const FLOAT	x_wall_st = 1.0;
				const FLOAT	x_wall_en = (nx-1);

				const FLOAT	x = 0.5 + 1.0*i;

				FLOAT	l;
				if (i < nx/2)	{ l = x_wall_st - x; }
				else			{ l = x - x_wall_en; }

				int		frg = STATUS_FLUID;
				if (l > 0.0) {
					frg = STATUS_WALL;
				}


				l_obs [id] = l;
				status[id] = frg;
			}
		}
	}
}

void Set_Obstacle_Sphere(MPIinfo *mpi, Domain *cdo, char *status, FLOAT *l_obs)
{
	const int	nx = cdo->nx,
				ny = cdo->ny,
				nz = cdo->nz;
	const FLOAT	dl = 1.0;

	// mpi
	const int	rank_x = mpi->rank_x,
				rank_y = mpi->rank_y,
				rank_z = mpi->rank_z;

	const int	nxg = cdo->nxg,
				nyg = cdo->nyg;
//				nzg = cdo->nzg;

	const FLOAT	xc = dl * nxg/2;
	const FLOAT	yc = dl * nyg/2;
//	const FLOAT	zc = dl * nzg/4;
	const FLOAT	zc = dl * nxg/2;
	const FLOAT	rc = dl * nxg/6;

	const int	Offset_X = 1; // バグを誘発する記述 //
	const int	Offset_Y = 1; // バグを誘発する記述 //
	const int	Offset_Z = 1;

	// init
	// p, T, u, v
	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				const int	ig = rank_x*(nx-2*Offset_X) + (i-Offset_X);
				const int	jg = rank_y*(ny-2*Offset_Y) + (j-Offset_Y);
				const int	kg = rank_z*(nz-2*Offset_Z) + (k-Offset_Z);

				const FLOAT	x = 0.5*dl + ig   *dl;
				const FLOAT	y = 0.5*dl + jg   *dl;
				const FLOAT	z = 0.5*dl + kg   *dl;

				// sphere
				const FLOAT	r = sqrt( pow(x-xc, 2) + pow(y-yc, 2) + pow(z-zc, 2) );

				int		frg = STATUS_FLUID;
//				if (r < rc)	{ frg = STATUS_WALL; }
				
				FLOAT	l = rc - r; // + wall
				if (l > 0.0)	{ frg = STATUS_WALL; }

				l_obs [id] = l;
				status[id] = frg;
			}
		}
	}
}


// with levelset
void Set_Obstacle_ground_surface ( 
	MPIinfo *mpi, Domain *cdo, char *status, FLOAT *l_obs)
{
	const int	nx = cdo->nx,
				ny = cdo->ny,
				nz = cdo->nz;

	FLOAT	ground = nx;

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				if (status[id] == STATUS_WALL && i <= ground) {
					ground = (FLOAT)i;
				}
			}
		}
	}
	ground += - 0.5;

	FLOAT	ground_g;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&ground,		&ground_g,		1, MFLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				const FLOAT	x = 0.5 + 1.0*i;

				if (x <= ground_g) {
					status[id] = STATUS_DIRICHLET_XM;
				}
			}
		}
	}

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				if (i == 0  || i == nx-1) { 
					status[id] = STATUS_SLIP_X;
			   	}
			}
		}
	}

//	for (int k=0; k<nz; k++) {
//		for (int j=0; j<ny; j++) {
//			for (int i=0; i<nx; i++) {
//				const int	id = i + nx*j + nx*ny*k;
//
//				const FLOAT	x = 0.5 + 1.0*i;
//
//				int		flag;
//
//				const FLOAT	lv_ground = ground_g - x;
//				const FLOAT	lv        = l_obs[id];
//
//				// 距離関数
//				FLOAT	sgn;
//				FLOAT	lv_min;
//				if (lv*lv_ground > 0.0) {
//					if (lv >= 0.0)	{ sgn =  1.0;	flag = STATUS_WALL ;	}
//					else			{ sgn = -1.0; 	flag = STATUS_FLUID;	}
//
//					lv_min = fmin(fabs(lv_ground), fabs(lv)) * sgn;
//				}
//				else {
//					// 物体優先(+側優先)
//					flag = STATUS_WALL;
//
//					if (lv > 0.0)	{ lv_min = lv; }
//					else			{ lv_min = lv_ground; }
//				}
//				l_obs [id] = lv_min;
//				status[id] = flag;
//			}
//		}
//	}

}


// Set_Obstacle.cu
