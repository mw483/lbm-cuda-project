#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>


// define //
#include "Define.h"
#include "defineMemory.h"
// define //


// parser //
#include "option_parser.h"
// parser //


// param //
#include "paramCalFlag.h"
#include "paramDomain.h"
#include "paramVariables.h"
#include "paramBasisVariables.h"
#include "paramFluidProperty.h"
#include "paramStress.h"
// param //


// output //
#include "outputStatus.h"
#include "outputParaview.h"
// output //

// Lib //
#include "functionLib.h"
// Lib //


// MPI //
#include "paramMPI.h"
// MPI //


// calculation //
#include "calculation.h"
// calculation //


/// output_user_def ///
#include "output_user_def.h"
/// output_user_def ///

/// output_user_result ///
#include "Output_user_results.h"
/// output_user_result ///

// (YOKOUCHI 2020)
// Define_user //
#include "Define_user.h"
//Define_user //

#ifdef PARTICLE_VISUALIZATION
  #include "paramParticle.h"

  #include "Cal_Particle.h"
  #include "Paraview_Particle.h"
  #include "ParamParticle_MPI.h"
  #include "MPI_Particle.h"
// (YOKOUCHI 2020)
  #include <curand_kernel.h>
#endif


using namespace std;


int main(int argc, char *argv[])
{
	char	*program_name = "LatticeBoltzmannMethod";
	enum defineMemory::FlagHostDevice	host_memory   = defineMemory::Host_Memory,
										device_memory = defineMemory::Device_Memory;

	// option parser //
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);

	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }
	// option parser //


	// CUDA device //
	int		ncpu, rank;
//    int  required = MPI_THREAD_FUNNELED;
//    int  provided;
//    MPI_Init_thread(&argc, &argv, required, &provided);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const int	gpu_node = parser.gpu_per_node();
	const int	device_num = (rank%gpu_node); // TSUBAME (number of gpus = 3)
	cudaSetDevice(device_num);
	// CUDA device //


	// MPI //
	paramMPI	pmpi (
		program_name,
		argc,
		argv
		);

	MPIinfo	mpi = pmpi.mpiinfo();
	if (rank == 0)	{ pmpi.cout_MPI (); }
	// MPI //


	// class : paramCalFlag //
	if (rank == 0)	{ std::cout << "paramCalFlag" << endl; }
	paramCalFlag pcalflg (
		program_name,
		argc,
		argv
		);
	if (rank == 0)	{ pcalflg.cout_CalFlag(); }

	CalFlag		calflg = pcalflg.calflg();
	if (rank == 0)	{ pcalflg.cout_CalFlag(calflg); }


	const int	cout_step    = pcalflg.cout_step(),
				fout_step    = pcalflg.fout_step();
	const int	cout_flg     = pcalflg.cout_flg(),
		  		fout_flg     = pcalflg.fout_flg(),
				restartf_flg = pcalflg.restart_out_flg();
	const int	pout_flg = parser.flag_particle_generate();
	int			fstep        = pcalflg.fstart();

	static bool	first_cal = 0;
	if (pcalflg.restart() == 1) { first_cal = 1; }
	// class : paramCalFlag //


	// class : paramDomain //
	if (rank == 0)	{ std::cout << "paramDomain" << endl; }
	paramDomain
		pdomain_h (
			program_name,
			argc,
			argv,
			mpi,
			host_memory
			);

	paramDomain
		pdomain_d (
			program_name,
			argc,
			argv,
			mpi,
			device_memory
			);

	pdomain_h.memcpy_paramDomain (
		pdomain_d,
		pdomain_h
		);

	Domain		domain_h = pdomain_h.domain();
	if (rank == 0)	{ pdomain_h.cout_Domain (); }

	const int	step  = pdomain_h.step();
	if (rank == 0)	{ std::cout << endl; }
	if (pcalflg.restart() == 1) { pdomain_h.read_Condition (); }
	// class : paramDomain //


	// MPI //
	pmpi.set_buffer (pdomain_h.domain());
	// MPI //


	// class : Variables //
	if (rank == 0)	{ std::cout << "paramVariables" << endl; }
	Variables	variables_h,
				variables_d, variablesn_d;

	paramVariables	pvariables (
			pmpi,
			pcalflg,
			pdomain_h
			);

	pvariables.allocate (
			&variables_h,
			host_memory);

	pvariables.allocate (
			&variables_d,
			device_memory);

	pvariables.allocate (
			&variablesn_d,
			device_memory);

	pvariables.init_host (&variables_h);


	pvariables.memcpy_Variables (&variables_d,  &variables_h);
	pvariables.memcpy_Variables (&variablesn_d, &variables_h);
	// class : Variables //


	// class : BasisVariables //
	if (rank == 0)	{ std::cout << "paramBasisVariables" << endl; }
	BasisVariables	cbq_h,
					cbq_d;

	paramBasisVariables	pbasisv (
			program_name,
			argc,
			argv,
			pmpi,
			pcalflg,
			pdomain_h);

	pbasisv.allocate (
			&cbq_h,
			host_memory);

	pbasisv.allocate (
			&cbq_d,
			device_memory);

	pbasisv.init_host (&cbq_h);

	pbasisv.memcpy_BasisVariables (&cbq_d,  &cbq_h);
	// class : BasisVariables //


	// class : paramStress //
	if (rank == 0)	{ std::cout << "paramStress" << endl; }
	Stress		str_h,
				str_d;

	paramStress	pstress (
			program_name,
			argc,
			argv,
			pdomain_h);

	pstress.allocate (
			&str_h,
			host_memory);

	pstress.allocate (
			&str_d,
			device_memory);

	pstress.init_host (&str_h);

	pstress.memcpy_Stress (&str_d,  &str_h);
	// class : paramStress //


	// class : paramFluidProperty //
	if (rank == 0)	{ std::cout << "paramFluidProperty" << endl; }
	FluidProperty	fluid_h,
					fluid_d;
	paramFluidProperty	pfluid (
							program_name,
							argc,
							argv,
							pmpi,
							pdomain_h);
MPI_Barrier(MPI_COMM_WORLD); // MOD 2018
if (rank == 0)  { std::cout << "pfluid.allocate host" << endl; }


	pfluid.allocate (
		&fluid_h,
		host_memory);

MPI_Barrier(MPI_COMM_WORLD); // MOD 2018
if (rank == 0)  { std::cout << "pfluid.allocate device" << endl; }

	pfluid.allocate (
		&fluid_d,
		device_memory);

MPI_Barrier(MPI_COMM_WORLD); // MOD 2018
if (rank == 0)  { std::cout << "pfluid.init_host" << endl; }

	pfluid.init_host        (&fluid_h);

if (rank == 0)  { std::cout << "read_global_map" << endl; }

	pfluid.read_global_map  (&fluid_h);

if (rank == 0)  { std::cout << "pfluid.memcpy" << endl; }

	pfluid.memcpy_FluidProperty (&fluid_d,  &fluid_h);
	// class : paramFluidProperty //

if (rank == 0)  { std::cout << "outputStatus pfluid" << endl; }

	// class : Status //
	outputStatus	status (
				program_name,
				argc,
				argv,
				mpi,
				calflg,
				pdomain_h.domain()
				);
	MPI_Barrier(MPI_COMM_WORLD);
	// class : Status //

if (rank == 0)  { std::cout << "Paraview" << endl; }

	// class : Paraview //
	Paraview	paraview (
					program_name,
					argc,
					argv,
					pmpi,
					pdomain_h);
	MPI_Barrier(MPI_COMM_WORLD);
	// class : Paraview //


	// calculation //
	Calculation	calculation (
					pmpi,
					pdomain_h);


	MPI_Barrier(MPI_COMM_WORLD);
	if (pcalflg.restart() == 0) {
		if (rank == 0) { cout << "Init_GPU_Cal \n"; }
		calculation.initialize_calculation (
		   	&variables_d,
		   	&variablesn_d,
		   	&cbq_d,
		   	&fluid_d
			);
	}
	// calculation //

	// Output_user_results //
	Output_user_results	 outRes( pmpi, pdomain_h, &variables_h, &cbq_h, &fluid_h, &str_h);
	// Output_user_results //

#ifdef PARTICLE_VISUALIZATION
	Particle		particle_h,
					particle_d;

	// class : paramParticle //
	paramParticle	paramparticle (
						program_name,
						argc,
						argv,
						rank,
						&particle_h,
						&particle_d);
	// class : paramParticle //

	CalParticle		calParticle (
						argc, argv,
						pmpi,
						pdomain_h,
						&particle_h);

/*	// (YOKOUCHI 2020)
	curandState		*state;
	if (user_flags::flg_particle == 1) {
		// make rundom number
		int nn = parser.particle();
		cudaMalloc((void**)&state, 	3 * nn * sizeof(curandState));
		setRandNum(3 * nn, particle_d.pgrid.parray_end, state);
	 
	}
*/	
	if (rank == 0) { cout << "init : particle ***** \n"; }

	Init_Particle_MPI (
		&mpi,
		&domain_h, particle_h.ppos, &particle_h.pmpi_host,
		&domain_h, particle_d.ppos, &particle_d.pmpi_host);

	if (particle_h.pfrag.prestart == 1) {
		paramparticle.Read_ParticlePosition(&particle_h, &particle_d);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	const int	pout_step = particle_h.pfrag.pout_step;
	if (rank == 0) { cout << "(pout_step) = (" << pout_step << ")\n"; }

	int		pstep = pcalflg.pstartstep();


	// paraview //
	const int	ncpu_div_p = parser.ncpu_div(3);
	ParaviewParticle	ParaviewParticle (rank, mpi.ncpu, ncpu_div_p, pdomain_h.halo(), pdomain_h.c_ref());
	// paraview //
#endif


	// time //
	struct	timeval	t1, t2;
	gettimeofday(&t1, NULL);
	// time //

                        output_user_def (
                                0,
                               program_name, argc, argv,
                                pmpi,
                               pdomain_h,
                                &variables_h,
                                &cbq_h,
                                &fluid_h
                                );
//						outRes.user_file_output(0);
	
			// (YOKOUCHI 2020) check vis_sgs //
/*			output_vis_sgs (
				0,
				program_name, argc, argv,
				pmpi,
				pdomain_h,
				&str_h,
				&fluid_h
				);
*/


	MPI_Barrier(MPI_COMM_WORLD);
	for (int i=0; i<step; i++) {
		if (rank == 0)	{ cout << "-" << flush; }


		// Cal //
		if (rank == 0 && i == 0)	{ cout << "calculation.GPU_Cal" << endl; }
		calculation.gpu_calculation (
			&variables_d,
			&variablesn_d,
			&cbq_d,
			&str_d,
			&fluid_d
			);
		// Cal //

		if (i%cout_step == 0) {
			if (rank == 0)	{ cout << "\n t / step = " << i << " / " << step << " , time = ( " << pdomain_h.time() << " ) " << "\n"; }
			if (cout_flg == 1 || fout_flg == 1 || restartf_flg == 1) {
				// memcopy //
				pvariables.memcpy_Variables (&variables_h,  &variables_d);
				// memcopy

				pbasisv.memcpy_BasisVariables (&cbq_h,    &cbq_d  );
				pfluid. memcpy_FluidProperty  (&fluid_h,  &fluid_d);
				pstress.memcpy_Stress         (&str_h,    &str_d  );
				// memcopy //
			}

			// check status
			if (cout_flg == 1) {
				status.cout_Status (
					cbq_h.r_n, cbq_h.u_n, cbq_h.v_n, cbq_h.w_n,
					fluid_h.vis_lbm, fluid_h.vis0_lbm, str_h.vis_sgs, str_h.Div,
					fluid_h.status
					);
			}

			if (i%fout_step == 0) {
				// restart
				if (restartf_flg == 1) {
					if (rank == 0)	{ cout << fstep << " : ----- restart output -----\n\n"; }

					pdomain_h.output_Condition ();

					pvariables.     output_restart_Variables      (&variables_h);
					pbasisv.output_restart_BasisVariables (&cbq_h);
				}

				if (fout_flg == 1) {
					// file output
					if (rank == 0)	{ cout << fstep << " : ----- file output -----\n"; }
					const int	clip_x = 0,
								clip_y = 0,
								clip_z = 0;
					const int	downsize1 = 1;
					const int	downsize2 = 2;
					const int	downsize4 = 4;

					// condition //


					// vtr header
//					const int	vtr_file_step = 0;
					const int	vtr_fstep = fstep;
					paraview.Make_vtr_binary_file(vtr_fstep, clip_x, clip_y, clip_z, downsize1);
					paraview.Make_vtr_binary_file(vtr_fstep, clip_x, clip_y, clip_z, downsize2);
					paraview.Make_vtr_binary_file(vtr_fstep, clip_x, clip_y, clip_z, downsize4);

					paraview.Make_multiblock_dataset(vtr_fstep, clip_x, clip_y, clip_z, downsize1);
					paraview.Make_multiblock_dataset(vtr_fstep, clip_x, clip_y, clip_z, downsize2);
					paraview.Make_multiblock_dataset(vtr_fstep, clip_x, clip_y, clip_z, downsize4);


					// binary output
					paraview.Output_Fluid_binary_All(vtr_fstep, &variables_h, &cbq_h, &fluid_h, &str_h, downsize1);
					paraview.Output_Fluid_binary_All(vtr_fstep, &variables_h, &cbq_h, &fluid_h, &str_h, downsize2);
					paraview.Output_Fluid_binary_All(vtr_fstep, &variables_h, &cbq_h, &fluid_h, &str_h, downsize4);


					// vtr convert
					paraview.Output_Fluid_vtr_binary_All(vtr_fstep, downsize1);
					paraview.Output_Fluid_vtr_binary_All(vtr_fstep, downsize2);
					paraview.Output_Fluid_vtr_binary_All(vtr_fstep, downsize4);


				}
				if (rank == 0)	{ cout << "\n"; }
			}
		}

#if 1
		/// output user def ///
		if (i%cout_step == 0) {
		output_user_def (
				i,
				program_name, argc, argv,
				pmpi,
				pdomain_h,
				&variables_h,
				&cbq_h,
				&fluid_h
				);
			//outRes.user_file_output(i);
		
			// (YOKOUCHI 2020) check vis_sgs //
		/*if (user_flags::flg_particle == 1) {
			output_vis_sgs (
					i,
					program_name, argc, argv,
					pmpi,
					pdomain_h,
					&str_h,
					&cbq_h,
					&fluid_h
					);
		*/
		}
		/// output user def ///
#endif

		if (i%fout_step == 0)		{ fstep++; }


#ifdef PARTICLE_VISUALIZATION
		if (rank == 0 && i == 0)	{ cout << "calParticle.particleAdvection" << endl; }
		
		// (YOKOUCHI 2020)
		if (user_flags::flg_particle == 1) {
			// cal mean velocity
			calculation.mean_velocity (
				&cbq_d,
				&str_d,
				i);
				
			// cal sgs tke
			calculation.sgs_tke_LBM (
				&cbq_d,
				&str_d,
				i);
			
			calParticle.particleAdvection_LSM (
				argc, argv,
				i,
			   	domain_h,
		   		&particle_h, &particle_d,
		   		&cbq_d,
		   		&fluid_d,
				&str_d);
		} else {
			calParticle.particleAdvection (
				argc, argv,
				i,
			   	domain_h,
			   	&particle_h, &particle_d,
			   	&cbq_d,
			   	&fluid_d);
		}

		if (i%pout_step == 0) {
			// variables
			pbasisv.memcpy_BasisVariables(&cbq_h,  &cbq_d);
			pstress.memcpy_Stress(&str_h, &str_d);

			// particle
			paramparticle.cudaMemcpy_ParticleGrid     (particle_h.pgrid, particle_d.pgrid);
			paramparticle.cudaMemcpy_ParticlePosition (particle_h.pgrid.parray_end, particle_h.ppos,  particle_d.ppos);

			// number of particles
			int		pnum_max = 0;
			MPI_Reduce(&particle_h.pgrid.parray_end,		&pnum_max,		1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0)	{
				cout << pstep << " : particle output : "
				     << pnum_max   << " / " << particle_h.pgrid.num_particle_max << " ( "
					 << (FLOAT)pnum_max/particle_h.pgrid.num_particle_max*100 << " % ) \n";
			}

			if (pout_flg == 1) {
				// (YOKOUCHI 2020)
				if (user_flags::flg_particle == 1 || user_flags::flg_particle == 2) {	
					ParaviewParticle.output_particle_binary_scatter_all_LSM (
						pstep,
						&domain_h,
						particle_h.pgrid.parray_end,
						particle_h.ppos,
						&cbq_h,
						&str_h,
						1);
				} else {
					ParaviewParticle.output_particle_binary_scatter_all (
						pstep,
						&domain_h,
						particle_h.pgrid.parray_end,
						particle_h.ppos,
						&cbq_h,
						&str_h,
						1);
				}
			}
		}
		if (i%pout_step == 0)	{ pstep++; }
#endif


		// first_cal //
		if (first_cal == 0)	{ first_cal = 1; }
		pdomain_h.time_evolution ();

		MPI_Barrier(MPI_COMM_WORLD);
	} // for ( time )
	gettimeofday(&t2, NULL);

	if (rank == 0) { printf("Elapsed time: %f\n", functionLib::get_elapsed_time(&t1, &t2)); }

	// mpi *****
	MPI_Finalize();
	// mpi *****

	return EXIT_SUCCESS;
}


// main.cu *****
