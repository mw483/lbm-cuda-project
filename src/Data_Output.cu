#include "Paraview.h"


// paraview
void Output_Binary_file(
	MPI_Library *mpi, 
	int t, Domain *cdo,
	const BasisVariables *cbq)
{
	const int	nx  = cdo->nx,
				ny  = cdo->ny,
				nz  = cdo->nz;
	const int	nxg = cdo->nxg,
				nyg = cdo->nyg,
				nzg = cdo->nzg;

	float	*f;
	f = new float [
}


void Paraview::Make_vtr_binary(MPI_Library *mpi, int t, Domain *cdo)
{
	if (mpi->rank == 0) {
		const int	buff = 1;

		const int	nx0 = cdo->nx,
					ny0 = cdo->ny,
					nz0 = cdo->nz;
		const int	ncpu_y = mpi->ncpu_y,
					ncpu_z = mpi->ncpu_z;
		const int	downsize[2] = { 1, 2 };

		FILE	*fp;
		char	fname[2][256];
		sprintf(fname[0], "./result/parallel-vector%d.pvtr",          t);
		sprintf(fname[1], "./result_downsize/parallel-vector%d.pvtr", t);

		for (int s=0; s<2; s++) {
			fp = fopen(fname[s], "w");

			const int	nx = (nx0-2*buff)/downsize[s] + 2*buff;
			const int	ny = (ny0-2*buff)/downsize[s] + 2*buff;
			const int	nz = (nz0-2*buff)/downsize[s] + 2*buff;

			fprintf(fp,"<?xml version=\"1.0\"?>\n");
			fprintf(fp,"<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

			fprintf(fp,"<PRectilinearGrid WholeExtent= \" %10d %10d %10d %10d %10d %10d \" GhostLevel=\"1\">\n", 0, nx-2, 0, ncpu_y*(ny-2*Offset_Y), 0, ncpu_z*(nz-2*Offset_Z));

			// cell data *****
			fprintf(fp,"<PCellData>\n");

//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"density\"/>\n");
			fprintf(fp,"<PDataArray type=\"Int8\"    format=\"appended\" Name=\"obstacle\"/>\n");
			fprintf(fp,"<PDataArray type=\"Int8\"    format=\"appended\" NumberOfComponents=\"3\" Name=\"velocity\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" NumberOfComponents=\"3\" Name=\"velocity\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"ss\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"ww\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"Q\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"E\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"vis_sgs\"/>\n");
//			fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"Fcs\"/>\n");

			fprintf(fp,"</PCellData>\n");
			// cell data *****

			// coordinates *****
			fprintf(fp,"<PCoordinates>\n");

			fprintf(fp,"<PDataArray type=\"Float32\" Name=\"x\" />\n");
			fprintf(fp,"<PDataArray type=\"Float32\" Name=\"y\" />\n");
			fprintf(fp,"<PDataArray type=\"Float32\" Name=\"z\" />\n");

			fprintf(fp,"</PCoordinates>\n");
			// coordinates *****

			// parallel datafile *****
			for (int k=0; k<ncpu_z; k++) {
				for (int j=0; j<ncpu_y; j++) {
					const int	id = j + ncpu_y * k;

					const int	nys = j *   (ny - 2*Offset_Y);
					const int	nye = nys + (ny - 2*Offset_Y);

					const int	nzs = k *   (nz - 2*Offset_Z);
					const int	nze = nzs + (nz - 2*Offset_Z);

					fprintf(fp,"<Piece Extent= \" %10d %10d %10d %10d %10d %10d \" Source=\"./vector%d-%d.vtr\"/>\n", 0, nx-2, nys, nye, nzs, nze, id, t);
				}
			}
			// parallel datafile *****

			fprintf(fp,"</PRectilinearGrid> </VTKFile>\n");

			// file close
			fclose(fp);
		}
	}
}


void Paraview::Output_Fluid_vtr_binary(MPI_Library *mpi, int t, Domain *cdo,
	const FLOAT *r, const FLOAT *u, const FLOAT *v, const FLOAT *w, 
	const FLOAT *SS, const FLOAT *WW, const char *frg,
	const FLOAT *vis_sgs, const FLOAT *Fcs)
{
	const int	buff = 1;

	const int	nx0 = cdo->nx,
				ny0 = cdo->ny,
				nz0 = cdo->nz;
	const int	nl0 = cdo->n0;
	const int	downsize[2] = { 1, 2 };

	const int	rank   = mpi->rank,
				rank_y = mpi->rank_y,
		  		rank_z = mpi->rank_z;

	FILE	*fp;
	char	fname[2][256];
	sprintf(fname[0], "./result/vector%d-%d.vtr",          rank, t);
	sprintf(fname[1], "./result_downsize/vector%d-%d.vtr", rank, t);

	for (int s=0; s<2; s++) {
		fp = fopen(fname[s], "w");

		const int	nx = (nx0-2*buff)/downsize[s] + 2*buff;
		const int	ny = (ny0-2*buff)/downsize[s] + 2*buff;
		const int	nz = (nz0-2*buff)/downsize[s] + 2*buff;
		const int	nl = (nl0-2*buff)/downsize[s] + 2*buff;

		const FLOAT	dl_out = L_REF / nl;

		const int	nys = mpi->rank_y * (ny - 2*buff),
					nye = nys + (ny - 2*buff),
					nzs = mpi->rank_z * (nz - 2*buff),
					nze = nzs + (nz - 2*buff);

		fprintf(fp,"<?xml version=\"1.0\"?>\n");
		fprintf(fp,"<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
		fprintf(fp,"<RectilinearGrid WholeExtent= \" %10d %10d %10d %10d %10d %10d \">\n", 0, nx-2*buff, nys, nye, nzs, nze);
		fprintf(fp,"<Piece Extent= \" %10d %10d %10d %10d %10d %10d \">\n", 0, nx-2*buff, nys, nye, nzs, nze);

		// coordinates *****
		fprintf(fp,"<Coordinates>\n");
		fprintf(fp,"<DataArray type=\"Float32\" Name=\"x\" format=\"ascii\">\n");
		for (int i=1; i<nx; i++) fprintf(fp, "%15f", (i-1)*dl_out);
		fprintf(fp,"</DataArray>\n");
	
		fprintf(fp,"<DataArray type=\"Float32\" Name=\"y\" format=\"ascii\">\n");
		for (int j=1; j<ny; j++) fprintf(fp, "%15f", (rank_y*(ny-2) + (j-1))*dl_out);
		fprintf(fp,"</DataArray>\n");
	
		fprintf(fp,"<DataArray type=\"Float32\" Name=\"z\" format=\"ascii\">\n");
		for (int k=1; k<nz; k++) fprintf(fp, "%15f", (rank_z*(nz-2) + (k-1))*dl_out);
		fprintf(fp,"</DataArray>\n");
	
		fprintf(fp,"</Coordinates>\n");
		// coordinates *****

		const uint32_t	size_n_i8 = (nx-2)*(ny-2)*(nz-2)*sizeof(int8_t);
//		const uint32_t	size_n_i  = (nx-2)*(ny-2)*(nz-2)*sizeof(int);
		const uint32_t	size_n_f  = (nx-2)*(ny-2)*(nz-2)*sizeof(float);
		int		offset = 0;
	
		// celldata *****
		fprintf(fp,"<CellData>\n");

//		// density
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"density\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);

		// obstacle
		fprintf(fp,"<DataArray type=\"Int8\" Name=\"obstacle\" format=\"appended\" offset=\"%d\" />\n", offset);
		offset += (4 + size_n_i8);
//		fprintf(fp,"<DataArray type=\"Int32\" Name=\"obstacle\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_i);

		// vector
//		fprintf(fp,"<DataArray type=\"Int8\" NumberOfComponents=\"3\" Name=\"velocity\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_i8*3);
		fprintf(fp,"<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"velocity\" format=\"appended\" offset=\"%d\" />\n", offset);
		offset += (4 + size_n_f*3);

//		// SS
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"SS\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);
//
//		// WW
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"WW\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);
//
//		// Q
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"Q\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);
//
//		// E
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"E\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);

//		// vis_sgs
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"vis_sgs\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);
//
//		// Fcs
//		fprintf(fp,"<DataArray type=\"Float32\" Name=\"Fcs\" format=\"appended\" offset=\"%d\" />\n", offset);
//		offset += (4 + size_n_f);


		FLOAT	*Q, *E;
		Q  = new FLOAT[nx*ny*nz];
		E  = new FLOAT[nx*ny*nz];

		for (int i=0; i<nx*ny*nz; i++) {
			Q[i] = WW[i] - SS[i];
			E[i] = WW[i] + SS[i];
		}

		fprintf(fp,"</CellData>\n");
		// celldata *****

		// end
		fprintf(fp,"</Piece>\n");
		fprintf(fp,"</RectilinearGrid>\n");

		// binary file
		fprintf(fp,"<AppendedData encoding=\"raw\">\n");
		fprintf(fp, "_");

		// density
//		Write_Scalar_binary(fp, buff, nx0, ny0, nz0, r, downsize[s]); 

		// obstacle
		Write_Scalar_Obstacle_int8_binary(fp, buff, nx0, ny0, nz0, frg, downsize[s]); 

		// vector
//		Write_Vector_int8_binary(fp, buff, nx0, ny0, nz0, u, v, w, downsize[s]);
		Write_Vector_binary(fp, buff, nx0, ny0, nz0, u, v, w, downsize[s]);

		// binary file end
		fprintf(fp,"</AppendedData>\n");
		fprintf(fp,"</VTKFile>\n");

		// file close
		fclose(fp);

		delete [] E;
		delete [] Q;
	}
}

// binary
void Paraview::Write_Scalar_int_binary(FILE *fp, int buff, int nx, int ny, int nz, const int *value, int downsize)
{
	const int		nxy = nx*ny;
	const int		size_format = sizeof(int);
	const uint32_t	size_n = ((nx-2*buff)/downsize)*((ny-2*buff)/downsize)*((nz-2*buff)/downsize)*size_format;
	int		tmp;

	fwrite(&size_n, sizeof(uint32_t), 1, fp);

	for (int k=buff; k<nz-buff; k+=downsize) {
		for (int j=buff; j<ny-buff; j+=downsize) {
			for (int i=buff; i<nx-buff; i+=downsize) {
				const int	id = i + nx*j + nxy*k;

				tmp = value[id];
				fwrite(&tmp, size_format, 1, fp);
			}
		}
	}
}

void Paraview::Write_Scalar_binary(FILE *fp, int buff, int nx, int ny, int nz, const FLOAT *value, int downsize)
{
	const int		nxy = nx*ny;
	const int		size_format = sizeof(float);
	const uint32_t	size_n = ((nx-2*buff)/downsize)*((ny-2*buff)/downsize)*((nz-2*buff)/downsize)*size_format;
	float	tmp;

	fwrite(&size_n, sizeof(uint32_t), 1, fp);

	for (int k=buff; k<nz-buff; k+=downsize) {
		for (int j=buff; j<ny-buff; j+=downsize) {
			for (int i=buff; i<nx-buff; i+=downsize) {
				const int	id = i + nx*j + nxy*k;

				tmp = (float)value[id];
				fwrite(&tmp, size_format, 1, fp);
			}
		}
	}
}

void Paraview::Write_Scalar_Obstacle_int8_binary(FILE *fp, int buff, int nx, int ny, int nz, const char *value, int downsize)
{
	const int		nxy = nx*ny;
	const int		size_format = sizeof(int8_t);
	const uint32_t	size_n = ((nx-2*buff)/downsize)*((ny-2*buff)/downsize)*((nz-2*buff)/downsize)*size_format;
	int		tmp;
	uint8_t	frg;

	fwrite(&size_n, sizeof(uint32_t), 1, fp);

	for (int k=buff; k<nz-buff; k+=downsize) {
		for (int j=buff; j<ny-buff; j+=downsize) {
			for (int i=buff; i<nx-buff; i+=downsize) {
				const int	id = i + nx*j + nxy*k;

				tmp = STATUS_WALL;
				if (value[id] != STATUS_WALL) { tmp = STATUS_FLUID; }

				frg = (int8_t)tmp;

				fwrite(&frg, size_format, 1, fp);
			}
		}
	}
}

void Paraview::Write_Vector_binary(FILE *fp, int buff, int nx, int ny, int nz, const FLOAT *u, const FLOAT *v, const FLOAT *w, int downsize)
{
	const int		nxy = nx*ny;
	const int		size_format = sizeof(float);
	const uint32_t	size_n = 3*((nx-2*buff)/downsize)*((ny-2*buff)/downsize)*((nz-2*buff)/downsize)*size_format;
	float	tmp_u, tmp_v, tmp_w;

	fwrite(&size_n, sizeof(uint32_t), 1, fp);

	for (int k=buff; k<nz-buff; k+=downsize) {
		for (int j=buff; j<ny-buff; j+=downsize) {
			for (int i=buff; i<nx-buff; i+=downsize) {
				const int	id = i + nx*j + nxy*k;

				tmp_u = (u[id]);
				tmp_v = (v[id]);
				tmp_w = (w[id]);

//				tmp_u = 0.5*(u[id] + u[id+1]);
//				tmp_v = 0.5*(v[id] + v[id+nx]);
//				tmp_w = 0.5*(w[id] + w[id+nxy]);

				tmp_u *= C_REF;
				tmp_v *= C_REF;
				tmp_w *= C_REF;

				fwrite(&tmp_u, size_format, 1, fp);
				fwrite(&tmp_v, size_format, 1, fp);
				fwrite(&tmp_w, size_format, 1, fp);
			}
		}
	}
}

void Paraview::Write_Vector_int8_binary(FILE *fp, int buff, int nx, int ny, int nz, const FLOAT *u, const FLOAT *v, const FLOAT *w, int downsize)
{
	const int		nxy = nx*ny;
	const int		size_format = sizeof(int8_t);
	const uint32_t	size_n = 3*((nx-2*buff)/downsize)*((ny-2*buff)/downsize)*((nz-2*buff)/downsize)*size_format;
	int8_t		uu, vv, ww;
	FLOAT		tmp_u, tmp_v, tmp_w;
	FLOAT		vel_min = -0.9,
				vel_max =  0.9,
				vel_range = fabs(vel_max - vel_min);

	fwrite(&size_n, sizeof(uint32_t), 1, fp);

	for (int k=buff; k<nz-buff; k+=downsize) {
		for (int j=buff; j<ny-buff; j+=downsize) {
			for (int i=buff; i<nx-buff; i+=downsize) {
				const int	id = i + nx*j + nxy*k;

				tmp_u = u[id];
				tmp_v = v[id];
				tmp_w = w[id];

				tmp_u = fmax(tmp_u, vel_min);	tmp_u = fmin(tmp_u, vel_max);
				tmp_v = fmax(tmp_v, vel_min);	tmp_v = fmin(tmp_v, vel_max);
				tmp_w = fmax(tmp_w, vel_min);	tmp_w = fmin(tmp_w, vel_max);

				tmp_u *= (1.0/vel_range) * 255;
				tmp_v *= (1.0/vel_range) * 255;
				tmp_w *= (1.0/vel_range) * 255;

				uu = (int8_t)(tmp_u);
				vv = (int8_t)(tmp_v);
				ww = (int8_t)(tmp_w);

//				std::cout << "uu, vv, ww = " << tmp_u << ", " << tmp_v << ", " << tmp_w << "\n";	getchar();
//				std::cout << "uu, vv, ww = " << uu << ", " << vv << ", " << ww << "\n";	getchar();

				fwrite(&uu, size_format, 1, fp);
				fwrite(&vv, size_format, 1, fp);
				fwrite(&ww, size_format, 1, fp);
			}
		}
	}
}


// Paraview.cu

