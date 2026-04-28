#include "outputParaview.h"

#include <iostream>
#include "option_parser.h"
#include "functionLib.h"
#include "fileLib.h"
#include "defineCoefficient.h"
#include "defineReferenceVel.h"
#include "defineBoundaryFlag.h"


void	Paraview::
set (
	char				*program_name,
	int					argc,
	char				*argv[], 
	const paramMPI		&pmpi, 
	const paramDomain	&pdomain
	)
{
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);

	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// mpi //
	rank_   = pmpi.rank();
	rank_x_ = pmpi.rank_x();
	rank_y_ = pmpi.rank_y();
	rank_z_ = pmpi.rank_z();

	ncpu_   = pmpi.ncpu();
	ncpu_x_ = pmpi.ncpu_x();
	ncpu_y_ = pmpi.ncpu_y();
	ncpu_z_ = pmpi.ncpu_z();


	// domain //
	nx_ = pdomain.nx();
	ny_ = pdomain.ny();
	nz_ = pdomain.nz();
	nn_ = pdomain.nn();

	n0_ = pdomain.n0();

	nxg_ = pdomain.nxg();
	nyg_ = pdomain.nyg();
	nzg_ = pdomain.nzg();

	halo_ = pdomain.halo();


	xg_min_ = pdomain.xg_min();
	yg_min_ = pdomain.yg_min();
	zg_min_ = pdomain.zg_min();

	dx_ = pdomain.dx();

	c_ref_ = pdomain.c_ref();

	// ncpu //
	ncpu_div_x_ = parser.ncpu_div(0);
	ncpu_div_y_ = parser.ncpu_div(1);
	ncpu_div_z_ = parser.ncpu_div(2);

	bnx_ = ncpu_x_ / ncpu_div_x_;
	bny_ = ncpu_y_ / ncpu_div_y_;
	bnz_ = ncpu_z_ / ncpu_div_z_;


	lnx_ = nxg_ / bnx_; 
	lny_ = nyg_ / bny_;
	lnz_ = nzg_ / bnz_;
	lnn_ = lnx_*lny_*lnz_;


	// check //
	check_ncpu_div ();
}


// make header file //
void	Paraview::
Make_vtr_binary_file(
	int		t,
	int		clip_x,
	int		clip_y,
	int		clip_z,
	int		downsize
	)
{
	if (rank_ == 0) {
		const int	bnx = bnx_,
					bny = bny_,
			  		bnz = bnz_;

		const int	lnx_out = lnx_ / downsize, 
					lny_out = lny_ / downsize, // = (ny - 2*Offset_Y) * ncpu_div_y_
					lnz_out = lnz_ / downsize;

		char	fname_folder_vtr[256];
		if      (downsize == 1)	{ sprintf(fname_folder_vtr, "./result"          ); }
		else if (downsize == 2)	{ sprintf(fname_folder_vtr, "./result_downsize2"); }
		else					{ sprintf(fname_folder_vtr, "./result_downsize" ); }

		char	fname[256];
		sprintf(fname, "%s/parallel-vector-x%d_y%d_z%d-%d.pvtr", fname_folder_vtr, clip_x, clip_y, clip_z, t);


		// fp //
		FILE	*fp = fopen(fname, "w");

		fprintf(fp,"<?xml version=\"1.0\"?>\n");
		fprintf(fp,"<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

		const int	nxs = lnx_out*       clip_x;
		const int	nxe = lnx_out*(bnx - clip_x);

		const int	nys = lny_out*       clip_y;
		const int	nye = lny_out*(bny - clip_y);

		const int	nzs = lnz_out*       clip_z;
		const int	nze = lnz_out*(bnz - clip_z);

		fprintf(fp,"<PRectilinearGrid WholeExtent= \" %10d %10d %10d %10d %10d %10d \" GhostLevel=\"1\">\n", nxs, nxe, nys, nye, nzs, nze);

		// cell data //
		fprintf(fp,"<PCellData>\n");

//		fprintf(fp,"<PDataArray type=\"Uint32\"  format=\"appended\" Name=\"id_stl\"/>\n");
		fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"levelset\"/>\n");
		fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"density\"/>\n");
		fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" Name=\"temperature\"/>\n");
		fprintf(fp,"<PDataArray type=\"Float32\" format=\"appended\" NumberOfComponents=\"3\" Name=\"velocity\"/>\n");

		fprintf(fp,"</PCellData>\n");
		// cell data //

		// coordinates //
		fprintf(fp,"<PCoordinates>\n");

		fprintf(fp,"<PDataArray type=\"Float32\" Name=\"x\" />\n");
		fprintf(fp,"<PDataArray type=\"Float32\" Name=\"y\" />\n");
		fprintf(fp,"<PDataArray type=\"Float32\" Name=\"z\" />\n");

		fprintf(fp,"</PCoordinates>\n");
		// coordinates //

		// parallel datafile //
		for (int k=clip_z; k<bnz-clip_z; k++) {
			for (int j=clip_y; j<bny-clip_y; j++) {
				for (int i=clip_x; i<bnx-clip_x; i++) {
					const int	id = i + bnx*j + bnx*bny*k;

					const int	nxs_l = i *     lnx_out;
					const int	nxe_l = nxs_l + lnx_out;

					const int	nys_l = j *     lny_out;
					const int	nye_l = nys_l + lny_out;

					const int	nzs_l = k *     lnz_out;
					const int	nze_l = nzs_l + lnz_out;

					fprintf(fp,"<Piece Extent= \" %10d %10d %10d %10d %10d %10d \" Source=\"./vector%d-%d.vtr\"/>\n",
							nxs_l, nxe_l, nys_l, nye_l, nzs_l, nze_l, id, t);
				}
			}
		}
		// parallel datafile //

		fprintf(fp,"</PRectilinearGrid> </VTKFile>\n");

		// file close
		fclose(fp);
	}
}


void	Paraview::
Make_multiblock_dataset (
	int		t,
	int		clip_x,
	int		clip_y,
	int		clip_z,
	int		downsize
	)
{
	if (rank_ == 0) {
		const int	bnx = bnx_,
					bny = bny_,
			  		bnz = bnz_;

		char	fname[256];

		// praview vtkMultiBlockDataSet *****
		sprintf(fname, "./result_hybrid/multi-block-vector-%d.vtm", t);
		FILE	*fp = fopen(fname, "w");

		//	fprintf(fp,"<?xml version=\"1.0\"?>\n");
		fprintf(fp,"<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
		fprintf(fp,"<vtkMultiBlockDataSet>\n");

		for (int k=clip_z; k<bnz-clip_z; k++) {
			for (int j=clip_y; j<bny-clip_y; j++) {
				for (int i=clip_x; i<bnx-clip_x; i++) {
					const int	id = i + bnx*j + bnx*bny*k;

					sprintf(fname, "../result_downsize/vector%d-%d.vtr", id, t);
					fprintf(fp,"<DataSet index=\"%d\" file=\"%s\">\n",id, fname);

					fprintf(fp,"</DataSet>\n");
				}
			}
		}
		fprintf(fp,"</vtkMultiBlockDataSet>\n");
		fprintf(fp,"</VTKFile>\n");

		// file close
		fclose(fp);
		// praview vtkMultiBlockDataSet *****
	}
}


void	Paraview::
Output_Fluid_binary_All (
	int		t,
	const Variables			*const cq,
	const BasisVariables	*const cbq,
	const FluidProperty		*const cfp,
	const Stress			*const str,
	int		downsize
	)
{
	const int	nx = nx_,
				ny = ny_,
				nz = nz_;

	// rank id
	const int	lnx = lnx_, 
				lny = lny_,
		  		lnz = lnz_;

	const int	lid_x = rank_x_ / ncpu_div_x_,
		  		lid_y = rank_y_ / ncpu_div_y_,
		  		lid_z = rank_z_ / ncpu_div_z_;
	const int	lid   = lid_x + bnx_*lid_y + bnx_*bny_*lid_z;


	// file index
	const int	fid_x = rank_x_ % ncpu_div_x_,
		  		fid_y = rank_y_ % ncpu_div_y_,
		  		fid_z = rank_z_ % ncpu_div_z_;
	const int	fid   = fid_x + ncpu_div_x_*fid_y + ncpu_div_x_*ncpu_div_y_*fid_z;


	// file
	char	fname_folder[256];
	if      (downsize == 1)	{ sprintf(fname_folder  , "result_binary"); }
	else if (downsize == 2)	{ sprintf(fname_folder  , "result_binary_downsize2"); }
	else					{ sprintf(fname_folder  , "result_binary_downsize"); }

	// all header //
	if (rank_ == 0) {
		char	fname_all_header[256];
		sprintf(fname_all_header , "./%s/all_header.dat"   , fname_folder     );

		FILE	*fp_all_header = fopen(fname_all_header  , "w");

		fwrite(&bnx_, sizeof(int), 1, fp_all_header);
		fwrite(&bny_, sizeof(int), 1, fp_all_header);
		fwrite(&bnz_, sizeof(int), 1, fp_all_header);

		const int	lnx_out = lnx/downsize,
					lny_out = lny/downsize,
					lnz_out = lnz/downsize;

		fwrite(&lnx_out, sizeof(int), 1, fp_all_header);
		fwrite(&lny_out, sizeof(int), 1, fp_all_header);
		fwrite(&lnz_out, sizeof(int), 1, fp_all_header);

		fclose(fp_all_header  );
	}
	// all header //

	MPI_Barrier(MPI_COMM_WORLD);

	// header //
	if (fid == 0) {
		char	fname_header[256];
		sprintf(fname_header, "./%s/header%d.dat"     , fname_folder, lid);

		FILE	*fp_header = fopen(fname_header  , "w");
	
		const int	lnx_out = lnx/downsize,
					lny_out = lny/downsize,
					lnz_out = lnz/downsize;
	
		fwrite(&lnx_out, sizeof(int), 1, fp_header);
		fwrite(&lny_out, sizeof(int), 1, fp_header);
		fwrite(&lnz_out, sizeof(int), 1, fp_header);
	
		fclose(fp_header  );
	}
	// header //


	// x, y, z //
	if (fid == 0) {
		char	fname_x[256];
		char	fname_y[256];
		char	fname_z[256];
		sprintf(fname_x, "./%s/axisx%d.dat", fname_folder, lid);
		sprintf(fname_y, "./%s/axisy%d.dat", fname_folder, lid);
		sprintf(fname_z, "./%s/axisz%d.dat", fname_folder, lid);


		FILE	*fp_x = fopen(fname_x, "w");
		FILE	*fp_y = fopen(fname_y, "w");
		FILE	*fp_z = fopen(fname_z, "w");

		FLOAT	dx = dx_,
				dy = dx_,
				dz = dx_;
		for (int i=0; i<lnx+1; i+=downsize) {
			const float	x = ( i + lnx*lid_x ) * dx + xg_min_;

			fwrite(&x, sizeof(float), 1, fp_x);
		}
		for (int j=0; j<lny+1; j+=downsize) {
			const float	y = ( j + lny*lid_y ) * dy + yg_min_;

			fwrite(&y, sizeof(float), 1, fp_y);
		}
		for (int k=0; k<lnz+1; k+=downsize) {
			const float	z = ( k + lnz*lid_z ) * dz + zg_min_;

			fwrite(&z, sizeof(float), 1, fp_z);
		}

		fclose(fp_x);
		fclose(fp_y);
		fclose(fp_z);
	}
	// x, y, z //


	MPI_Barrier(MPI_COMM_WORLD);


	// data output //
	char	fname[256];

//	// id_stl //
//	sprintf(fname, "./%s/id_stl%d.dat", fname_folder, lid);
//	write_binary_file_int (	cfp->id_obs,
//						fname, 
//						fid, 
//						downsize,
//						nx, ny, nz, lnx, lny, lnz);
//	// id_stl //


	// levelset //
	sprintf(fname, "./%s/levelset%d.dat", fname_folder, lid);
	write_binary_file (	cfp->l_obs,
						fname, 
						fid, 
						downsize,
						nx, ny, nz, lnx, lny, lnz);
	// levelset //


	// density //
	sprintf(fname, "./%s/density%d.dat", fname_folder, lid);
	write_binary_file (	cbq->r_n,
						fname, 
						fid, 
						downsize,
						nx, ny, nz, lnx, lny, lnz);
	// density //


	// temperature //
	sprintf(fname, "./%s/temperature%d.dat", fname_folder, lid);
	write_binary_file (	cq->T_n,
						fname, 
						fid, 
						downsize,
						nx, ny, nz, lnx, lny, lnz);
	// temperature //


	// velocity //
	const FLOAT	c_ref = c_ref_;
	FLOAT	*u_out = new FLOAT[nx*ny*nz];
	FLOAT	*v_out = new FLOAT[nx*ny*nz];
	FLOAT	*w_out = new FLOAT[nx*ny*nz];
	for (int i=0; i<nx*ny*nz; i++) {
		u_out[i] = cbq->u_n[i] * c_ref;
		v_out[i] = cbq->v_n[i] * c_ref;
		w_out[i] = cbq->w_n[i] * c_ref;
	}

	sprintf(fname, "./%s/velocity%d.dat", fname_folder, lid);
//	write_binary_file (	cbq->u_n, cbq->v_n, cbq->w_n,
	write_binary_file (	u_out, v_out, w_out,
						fname, 
						fid, 
						downsize,
						nx, ny, nz, lnx, lny, lnz);

	delete [] u_out;
	delete [] v_out;
	delete [] w_out;
	// velocity //


	MPI_Barrier(MPI_COMM_WORLD);
}


// vtr //
void	Paraview::
Output_Fluid_vtr_binary_All (
	int		t,
	int		downsize
	)
{
	// rank id
	const int	lid_x = rank_x_ / ncpu_div_x_,
		  		lid_y = rank_y_ / ncpu_div_y_,
		  		lid_z = rank_z_ / ncpu_div_z_;
	const int	lid   = lid_x + bnx_*lid_y + bnx_*bny_*lid_z;


	// file index
	const int	fid_x = rank_x_ % ncpu_div_x_,
		  		fid_y = rank_y_ % ncpu_div_y_,
		  		fid_z = rank_z_ % ncpu_div_z_;
	const int	fid   = fid_x + ncpu_div_x_*fid_y + ncpu_div_x_*ncpu_div_y_*fid_z;

	if (fid != 0) { return; }

	// file
	char	fname_header[256],
			fname_x[256], fname_y[256], fname_z[256];

	char	fname_folder[256];
	if      (downsize == 1)	{ sprintf(fname_folder  , "result_binary"); }
	else if (downsize == 2)	{ sprintf(fname_folder  , "result_binary_downsize2"); }
	else					{ sprintf(fname_folder  , "result_binary_downsize"); }

	sprintf(fname_header    , "./%s/header%d.dat"  , fname_folder, lid);
	sprintf(fname_x         , "./%s/axisx%d.dat"   , fname_folder, lid);
	sprintf(fname_y         , "./%s/axisy%d.dat"   , fname_folder, lid);
	sprintf(fname_z         , "./%s/axisz%d.dat"   , fname_folder, lid);


	// header //
	FILE	*fp_header     = fopen(fname_header      , "r");

	int		lnx_out, lny_out, lnz_out;
	fread(&lnx_out, sizeof(int), 1, fp_header);
	fread(&lny_out, sizeof(int), 1, fp_header);
	fread(&lnz_out, sizeof(int), 1, fp_header);

	fclose(fp_header  );
	// header //


	// write //
	char	fname_folder_vtr[256];
	if      (downsize == 1)	{ sprintf(fname_folder_vtr, "./result"          ); }
	else if (downsize == 2)	{ sprintf(fname_folder_vtr, "./result_downsize2"); }
	else					{ sprintf(fname_folder_vtr, "./result_downsize" ); }

	char	fname[256];
	if      (downsize == 1)	{ sprintf(fname, "%s/vector%d-%d.vtr", fname_folder_vtr, lid, t); }
	else if (downsize == 2)	{ sprintf(fname, "%s/vector%d-%d.vtr", fname_folder_vtr, lid, t); }
	else					{ sprintf(fname, "%s/vector%d-%d.vtr", fname_folder_vtr, lid, t); }


	FILE	*fp = fopen(fname, "w");
	const int	nxs = lid_x * lnx_out,
				nxe = nxs   + lnx_out,
				nys = lid_y * lny_out,
				nye = nys   + lny_out,
				nzs = lid_z * lnz_out,
				nze = nzs   + lnz_out;

	fprintf(fp,"<?xml version=\"1.0\"?>\n");
	fprintf(fp,"<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf(fp,"<RectilinearGrid WholeExtent= \" %10d %10d %10d %10d %10d %10d \">\n", nxs, nxe, nys, nye, nzs, nze);
	fprintf(fp,"<Piece Extent= \" %10d %10d %10d %10d %10d %10d \">\n", nxs, nxe, nys, nye, nzs, nze);


	// coordinates //
	FILE	*fp_x = fopen(fname_x, "r");
	FILE	*fp_y = fopen(fname_y, "r");
	FILE	*fp_z = fopen(fname_z, "r");

	float	*x = new float[lnx_out+1];
	float	*y = new float[lny_out+1];
	float	*z = new float[lnz_out+1];

	fread(x, sizeof(float), lnx_out+1, fp_x);
	fread(y, sizeof(float), lny_out+1, fp_y);
	fread(z, sizeof(float), lnz_out+1, fp_z);

	// vtr
	fprintf(fp,"<Coordinates>\n");
	fprintf(fp,"<DataArray type=\"Float32\" Name=\"x\" format=\"ascii\">\n");
	for (int i=0; i<lnx_out+1; i++) fprintf(fp, "%15f", x[i]);
	fprintf(fp,"</DataArray>\n");

	fprintf(fp,"<DataArray type=\"Float32\" Name=\"y\" format=\"ascii\">\n");
	for (int j=0; j<lny_out+1; j++) fprintf(fp, "%15f", y[j]);
	fprintf(fp,"</DataArray>\n");

	fprintf(fp,"<DataArray type=\"Float32\" Name=\"z\" format=\"ascii\">\n");
	for (int k=0; k<lnz_out+1; k++) fprintf(fp, "%15f", z[k]);
	fprintf(fp,"</DataArray>\n");
	
	fprintf(fp,"</Coordinates>\n");

	delete [] x;
	delete [] y;
	delete [] z;

	fclose(fp_x);
	fclose(fp_y);
	fclose(fp_z);
	// coordinates //


	const int		lnn_out = lnx_out*lny_out*lnz_out;
//	const uint32_t	size_n_i = lnn_out*sizeof(int);
	const uint32_t	size_n_f = lnn_out*sizeof(float);
	int		offset = 0;
	
	// celldata *****
	fprintf(fp,"<CellData>\n");


//	// id_stl //
//	fprintf(fp,"<DataArray type=\"Uint32\" Name=\"id_stl\" format=\"appended\" offset=\"%d\" />\n", offset);
//	offset += (4 + size_n_i);

	// levelset //
	fprintf(fp,"<DataArray type=\"Float32\" Name=\"levelset\" format=\"appended\" offset=\"%d\" />\n", offset);
	offset += (4 + size_n_f);

	// density //
	fprintf(fp,"<DataArray type=\"Float32\" Name=\"density\" format=\"appended\" offset=\"%d\" />\n", offset);
	offset += (4 + size_n_f);

	// temperature //
	fprintf(fp,"<DataArray type=\"Float32\" Name=\"temperature\" format=\"appended\" offset=\"%d\" />\n", offset);
	offset += (4 + size_n_f);

	// velocity
	fprintf(fp,"<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"velocity\" format=\"appended\" offset=\"%d\" />\n", offset);
	offset += (4 + size_n_f*3);


	fprintf(fp,"</CellData>\n");
	// celldata *****

	// end
	fprintf(fp,"</Piece>\n");
	fprintf(fp,"</RectilinearGrid>\n");

	// binary file
	fprintf(fp,"<AppendedData encoding=\"raw\">\n");
	fprintf(fp, "_");


	char	fname_dat[256];

//	// id_stl //
//	sprintf(fname_dat, "./%s/id_stl%d.dat", fname_folder, lid);
//	FILE	*fp_id_stl   = fopen(fname_dat, "r");
//	fileLib::fread_fwrite_data <int> (fp, fp_id_stl, lnn_out);
//	fclose(fp_id_stl);
//	// id_stl //

	// levelset //
	sprintf(fname_dat, "./%s/levelset%d.dat", fname_folder, lid);
	FILE	*fp_levelset   = fopen(fname_dat, "r");
	fileLib::fread_fwrite_data <float> (fp, fp_levelset, lnn_out);
	fclose(fp_levelset);
	// levelset //

	// density //
	sprintf(fname_dat, "./%s/density%d.dat", fname_folder, lid);
	FILE	*fp_density   = fopen(fname_dat, "r");
	fileLib::fread_fwrite_data <float> (fp, fp_density, lnn_out);
	fclose(fp_density);
	// density //

	// temperature //
	sprintf(fname_dat, "./%s/temperature%d.dat", fname_folder, lid);
	FILE	*fp_temperature   = fopen(fname_dat, "r");
	fileLib::fread_fwrite_data <float> (fp, fp_temperature, lnn_out);
	fclose(fp_temperature);
	// temperature //

	// velocity //
	sprintf(fname_dat, "./%s/velocity%d.dat", fname_folder, lid);
	FILE	*fp_velocity   = fopen(fname_dat, "r");
	fileLib::fread_fwrite_data <float> (fp, fp_velocity, 3*lnn_out);
	fclose(fp_velocity);
	// velocity //



	// binary file end
	fprintf(fp,"</AppendedData>\n");
	fprintf(fp,"</VTKFile>\n");


	// file close
	fclose(fp);
}


// private //
void Paraview::
check_ncpu_div ()
{
	if ( (ncpu_x_%ncpu_div_x_) != 0 
	  || (ncpu_y_%ncpu_div_y_) != 0 
	  || (ncpu_z_%ncpu_div_z_) != 0 ) {
		std::cout << "error : ncpu_div_x_, ncpu_div_y_, ncpu_div_z_\n";

		exit(2);
	}
}


void Paraview::
write_binary_file_int (
	int		*f,
	char	fname[],
	int		fid,
	int		downsize,
	int		nx,  int ny,  int nz,
	int		lnx, int lny, int lnz
	)
{
	const int	lnn = lnx*lny*lnz;

	int	*f_tmp = new int[lnn];
	functionLib::fillArray (f_tmp,  0, lnn);
	MPI_Barrier(MPI_COMM_WORLD);

	combine_data <int> (f_tmp,  lnx, lny, lnz, f,  nx, ny, nz);
	MPI_Barrier(MPI_COMM_WORLD);

	if (fid == 0) {
		FILE	*fp = fopen(fname, "w");
		const uint32_t	size_f   = lnn*sizeof(int) / (int)pow((int)downsize, 3);
		fwrite(&size_f, sizeof(uint32_t), 1, fp);

		for (int k=0; k<lnz; k+=downsize) {
			for (int j=0; j<lny; j+=downsize) {
				for (int i=0; i<lnx; i+=downsize) {
					const int	id = i + lnx*j + lnx*lny*k;

					int	tmp = (int)f_tmp[id];

					fwrite(&tmp, sizeof(int), 1, fp);
				}
			}
		}
		fclose(fp);
	}
	delete [] f_tmp;
}


void Paraview::
write_binary_file (
	FLOAT	*f,
	char	fname[],
	int		fid,
	int		downsize,
	int		nx,  int ny,  int nz,
	int		lnx, int lny, int lnz
	)
{
	const int	lnn = lnx*lny*lnz;

	FLOAT	*f_tmp = new FLOAT[lnn];
	functionLib::fillArray (f_tmp,  0.0, lnn);
	MPI_Barrier(MPI_COMM_WORLD);

	combine_data <FLOAT> (f_tmp,  lnx, lny, lnz, f,  nx, ny, nz);
	MPI_Barrier(MPI_COMM_WORLD);

	if (fid == 0) {
		FILE	*fp = fopen(fname, "w");
		const uint32_t	size_f   = lnn*sizeof(float) / (int)pow((float)downsize, 3);
		fwrite(&size_f, sizeof(uint32_t), 1, fp);

		for (int k=0; k<lnz; k+=downsize) {
			for (int j=0; j<lny; j+=downsize) {
				for (int i=0; i<lnx; i+=downsize) {
					const int	id = i + lnx*j + lnx*lny*k;

					float	tmp = f_tmp[id];

					fwrite(&tmp, sizeof(float), 1, fp);
				}
			}
		}
		fclose(fp);
	}
	delete [] f_tmp;
}


void Paraview::
write_binary_file (
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	char	fname[],
	int fid,
	int downsize,
	int nx,  int ny,  int nz,
	int lnx, int lny, int lnz)
{
	const int	lnn = lnx*lny*lnz;

	FLOAT	*u_tmp  = new FLOAT[lnn];
	FLOAT	*v_tmp  = new FLOAT[lnn];
	FLOAT	*w_tmp  = new FLOAT[lnn];
	functionLib::fillArray (u_tmp,  0.0, lnn);
	functionLib::fillArray (v_tmp,  0.0, lnn);
	functionLib::fillArray (w_tmp,  0.0, lnn);
	MPI_Barrier(MPI_COMM_WORLD);

	combine_data <FLOAT> (u_tmp,  lnx, lny, lnz, u, nx, ny, nz);
	combine_data <FLOAT> (v_tmp,  lnx, lny, lnz, v, nx, ny, nz);
	combine_data <FLOAT> (w_tmp,  lnx, lny, lnz, w, nx, ny, nz);
	MPI_Barrier(MPI_COMM_WORLD);


	if (fid == 0) {
		FILE	*fp_velocity = fopen(fname, "w");
		const uint32_t	size_vel   = 3*lnn*sizeof(float) / (int)pow((float)downsize, 3);
		fwrite(&size_vel, sizeof(uint32_t), 1, fp_velocity);

		for (int k=0; k<lnz; k+=downsize) {
			for (int j=0; j<lny; j+=downsize) {
				for (int i=0; i<lnx; i+=downsize) {
					const int	id = i + lnx*j + lnx*lny*k;

					float	vel[3] = { u_tmp[id], v_tmp[id], w_tmp[id] };

					fwrite(vel, sizeof(float), 3, fp_velocity);
				}
			}
		}
		fclose(fp_velocity);
	}

	delete [] u_tmp;
	delete [] v_tmp;
	delete [] w_tmp;
}


void Paraview::
Write_Scalar_Obstacle_int8_binary_file (
	FILE	*fp_write,
	FILE	*fp_read,
	int nx, int ny, int nz
	)
{
	const int		size_format = sizeof(int8_t);

	uint32_t	size;
	fread (&size, sizeof(uint32_t), 1, fp_read );
	fwrite(&size, sizeof(uint32_t), 1, fp_write);

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				uint8_t		tmp;
				fread (&tmp, sizeof(uint8_t), 1, fp_read );
				if (tmp != STATUS_WALL) { tmp = STATUS_FLUID; }
				
				uint8_t	frg = (uint8_t)tmp;

				fwrite(&frg, size_format, 1, fp_write);
			}
		}
	}
}


// Paraview.cu //
