#include "Generate_Particle.h"
#include "Define_user.h" // Read if user flag == 3

#include "macroCUDA.h"
#include "functionLib.h"
// read particle position (YOKOUCHI 2020)
#include <fstream>

void 
generate_particle (
	int		index_num,
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty
	)
{
	generate_particle_box    (index_num, domain, particle, fproperty);
//	generate_particle_sphere (index_num, domain, particle, fproperty);
}


void 
generate_particle_box (
	      int			index_num,
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty
	)
{
	// generate particles //
	const int	num_g[3] = {	
					1,
					10,
					10	
					};

	const FLOAT	point_g[3] = {
					domain.xg_min + domain.xg_length * 0.80/10.0,
					domain.yg_min + domain.yg_length * 3.75/10.0,
					domain.zg_min + domain.zg_length * 3.75/10.0
					};

	const FLOAT	vec_g[3] = {
					domain.xg_length * 3.0/10.0,
					domain.yg_length * 2.5/10.0,
					domain.zg_length * 2.5/10.0
					};	// vec_g > 0


	FLOAT	vec_dx[3];
	for (int i=0; i<3; i++) {
		if (num_g[i] == 1)	{ vec_dx[i] = 0.0; }
		else				{ vec_dx[i] = vec_g[i]/(num_g[i]-1); }
	}
 

	// filter (domain) //
	int		num  [3] = { num_g  [0], num_g  [1], num_g  [2] };
	FLOAT	point[3] = { point_g[0], point_g[1], point_g[2] };


	particle_source_box (
		domain, 
		particle->pfrag, 
		particle->pgrid, 
		particle->ppos, 
		num,
		point, 
		vec_dx,
		index_num
		);
}


void 
generate_particle_box_slice (
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty,
	const int			num_g  [],
	const FLOAT			point_g[],
	const FLOAT			vec_g  []
	)
{
//	const int	num_g[3] = {
//					1,
//					10,
//					10
//					};
//	const FLOAT	point_g[3] = {
//					domain.xg_min + domain.xg_length * 0.80/10.0,
//					domain.yg_min + domain.yg_length * 3.75/10.0,
//					domain.zg_min + domain.zg_length * 3.75/10.0
//					};
//	const FLOAT	vec_g[3] = {
//					domain.xg_length * 3.0/10.0,
//					domain.yg_length * 2.5/10.0,
//					domain.zg_length * 2.5/10.0
//					};	// vec_g > 0

	FLOAT	vec_dx0[3];
	for (int i=0; i<3; i++) {
		if (num_g[i] == 1)	{ vec_dx0[i] = 0.0; }
		else				{ vec_dx0[i] = vec_g[i]/(num_g[i]-1); }
	}


	      int	index_num = 0;
	const int	dim       = 2;

	for (int ii=0; ii<num_g[dim]; ii++) {
		// filter (domain) //
		int		num   [3] = { num_g  [0], num_g  [1], num_g  [2] };
		FLOAT	point [3] = { point_g[0], point_g[1], point_g[2] };
		FLOAT	vec_dx[3] = { vec_dx0[0], vec_dx0[1], vec_dx0[2] };

		// dim //
		num   [dim] = 1;
		vec_dx[dim] = 0.0;
		point [dim] = point_g[dim] + vec_dx0[dim]*ii;

//		int		num  [3] = { num_g  [0], num_g  [1], num_g  [2] };


		particle_source_box (
			domain,
			particle->pfrag, 
			particle->pgrid, 
			particle->ppos,
			num,
			point,
			vec_dx,
			index_num
			);
	}

}

// (YOKOUCHI 2020)
void
generate_particle_LSM (
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty,
	const int		num_g[],
	const FLOAT		point_g[],
	const FLOAT		vec_g[],
	int			t,
	int			pstart
	)
{
	FLOAT	vec_dx0[3];
	for (int i=0; i<3; i++) {
		if (num_g[i] == 1)	{ vec_dx0[i] = 0.0; }
		else				{ vec_dx0[i] = vec_g[i]/(num_g[i]-1); }
	}

	      int	index_num = 0;
	const int	dim       = 2;

	for (int ii=0; ii<num_g[dim]; ii++) {
		// filter (domain) //
		int		num   [3] = { num_g  [0], num_g  [1], num_g  [2] };
		FLOAT	point [3] = { point_g[0], point_g[1], point_g[2] };
		FLOAT	vec_dx[3] = { vec_dx0[0], vec_dx0[1], vec_dx0[2] };

		// dim //
		num   [dim] = 1;
		vec_dx[dim] = 0.0;
		point [dim] = point_g[dim] + vec_dx0[dim]*ii;

//		int		num  [3] = { num_g  [0], num_g  [1], num_g  [2] };


		particle_source_LSM (
			domain,
			particle->pfrag, 
			particle->pgrid, 
			particle->ppos,
			num,
			point,
			vec_dx,
			index_num,
			t,
			pstart
			);
	}
}
	

void 
generate_particle_sphere (
	int		index_num,
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty
	)
{
	// generate particles
	const int	num_g[2] = {	
					(int)( 8),		// theta0
					(int)( 8)
					};	// theta1

	const FLOAT	point_g[3] = {	
					domain.xg_min + domain.xg_length * 5.0/10.0,
					domain.yg_min + domain.yg_length * 5.0/10.0,
					domain.zg_min + domain.zg_length * 5.0/10.0
					};

	const FLOAT	radius = {	
					domain.zg_min + domain.zg_length * 5.0/10.0
					};

	const FLOAT	theta0[2] = { -2.5/5.0*M_PI, 0.0 },
		  		theta1[2] = {  0.0, 2.0*M_PI };

	int		num  [2] = { num_g  [0], num_g  [1] };
	FLOAT	point[3] = { point_g[0], point_g[1], point_g[2] };

	// generate particle
	particle_source_sphere (
		domain,
		particle->pfrag,
		particle->pgrid, 
		particle->ppos, 
		num,
		point,
		radius,
		theta0,
		theta1,
		index_num
		);
}


void
particle_source_box (
	const Domain			&domain,
	      ParticleCalFlag	&pfrag,
		  ParticleGrid		&pgrid,
	      ParticlePosition	*ppos,
	const int	num[],
	const FLOAT	point[],
	const FLOAT	vec_dx[],
	int			&index_num
	)
{
	const int	num_source  = num[0]*num[1]*num[2];

	const int	thread_length  = 256;
	const int	block_number   = num_source/thread_length + 1;

	dim3	grid,
			block;
	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length, 1, 1);


	int		parray_start = pgrid.parray_end;
	if (parray_start + num_source < pgrid.num_particle_max) {
		// source //
		gpu_particle_source_box <<< grid, block >>> (
			ppos,
			index_num,
			parray_start,
			num   [0], num   [1], num   [2], 
			point [0], point [1], point [2],
			vec_dx[0], vec_dx[1], vec_dx[2]
			);

		cudaThreadSynchronize(); // cudaThreadSynchronize //


		// region check //
		const FLOAT	x_range_min[3] = { domain.x_min, domain.y_min, domain.z_min };
		const FLOAT	x_range_max[3] = { domain.x_max, domain.y_max, domain.z_max };

		gpu_particle_region_check <<< grid, block >>> (
			ppos,
			parray_start,
			num[0], num[1], num[2], 
			x_range_min[0], x_range_min[1], x_range_min[2],
			x_range_max[0], x_range_max[1], x_range_max[2]);

		cudaThreadSynchronize(); // cudaThreadSynchronize //


		pgrid.parray_end += num_source;
		index_num++;
	}
	else {
		std::cout << "Particle_Source_Box\n";
		std::cout << "number of particles is too large\n";
		exit(3);
	}
}

// (YOKOUCHI 2020)
void
particle_source_LSM (
	const Domain			&domain,
	      ParticleCalFlag		&pfrag,
	      ParticleGrid		&pgrid,
	      ParticlePosition		*ppos,
	const int			num[],
	const FLOAT			point[],
	const FLOAT			vec_dx[],
	int				&index_num,
	int				t,
	int				pstart
	)
{
	const int	c_ref=domain.c_ref;	// MOD 2021
	const int	num_source  = num[0]*num[1]*num[2];
	const int	thread_length  = 256;
	const int	block_number   = num_source/thread_length + 1;

	dim3	grid,
			block;
	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length, 1, 1);

	int		parray_start = pgrid.parray_end;
	if (parray_start + num_source < pgrid.num_particle_max) {
		
		// host
		FLOAT source_xd_h[num_source];
		FLOAT source_yd_h[num_source];
		FLOAT source_zd_h[num_source];
		FLOAT vel_us_h[num_source];
		FLOAT vel_vs_h[num_source];
		FLOAT vel_ws_h[num_source];
		int   Group_h[num_source];
		int   pos_idd_h[num_source];

		// device
		FLOAT *source_xd_d;
		FLOAT *source_yd_d;
		FLOAT *source_zd_d;
		FLOAT *vel_us_d;
		FLOAT *vel_vs_d;
		FLOAT *vel_ws_d;
		int   *Group_d;
		int   *pos_idd_d;
		
		// Allocate device memory //
		cudaMalloc(&source_xd_d, num_source*sizeof(FLOAT));
		cudaMalloc(&source_yd_d, num_source*sizeof(FLOAT));
		cudaMalloc(&source_zd_d, num_source*sizeof(FLOAT));
		cudaMalloc(&vel_us_d, 	 num_source*sizeof(FLOAT));
		cudaMalloc(&vel_vs_d,	 num_source*sizeof(FLOAT));
		cudaMalloc(&vel_ws_d, 	 num_source*sizeof(FLOAT));
		cudaMalloc(&Group_d,     num_source*sizeof(int));
		cudaMalloc(&pos_idd_d,   num_source*sizeof(int));
		
		// Read particle data //
		std::fstream fin;
		fin.open("./particle_position/particle_position.txt");
		if (!fin) {std::cout << "File (particle_position.txt) is not opened" << std::endl;}
		for(int i=0; i<num_source; i++) {
			fin >> source_xd_h[i];
			fin >> source_yd_h[i];
			fin >> source_zd_h[i];
			fin >> vel_us_h[i];
			fin >> vel_vs_h[i];
			fin >> vel_ws_h[i];
			fin >> Group_h[i];
			fin >> pos_idd_h[i];
		}
		fin.close();

		// ---------------------------------------------------------
        // Mikael 2026: Mobile Source Waypoint Implementation
        // ---------------------------------------------------------
        if (user_flags::flg_particle == 3) {
            std::ifstream fwp("./particle_position/particle_waypoints.txt");
            if (!fwp) { std::cout << "Warning: particle_waypoints.txt not found!" << std::endl; }

            // Calculate elapsed physical time
            FLOAT dx = domain.dx;
            FLOAT c_ref = domain.c_ref;
            FLOAT dt = dx / c_ref;
            FLOAT elapsed_time = (t - pstart) * dt;

            for (int i = 0; i < num_source; i++) {
                int mode, num_wp;
                fwp >> mode >> num_wp;

                // Buffer for waypoint data (Assuming max 20 waypoints per path)
                float wp_t[20], wp_x[20], wp_y[20], wp_z[20];
                for (int j = 0; j < num_wp; j++) {
                    fwp >> wp_t[j] >> wp_x[j] >> wp_y[j] >> wp_z[j];
                }

                // If no valid waypoints or dummy waypoint, skip calculation
                if (num_wp <= 1) {
                    continue; 
                }

                float total_duration = wp_t[num_wp - 1];
                float t_eff = elapsed_time;

                // --- 1. Mode Calculation (Loop, PingPong, Once) ---
                if (total_duration > 0.0f) {
                    if (mode == 1) { // Ping-pong
                        int cycle = (int)(elapsed_time / total_duration);
                        t_eff = fmodf(elapsed_time, total_duration);
                        if (cycle % 2 != 0) {
                            t_eff = total_duration - t_eff; // Reverse direction
                        }
                    } 
                    else if (mode == 2) { // Loop (Circuit)
                        t_eff = fmodf(elapsed_time, total_duration);
                    } 
                    else { // Once (Stop at end)
                        if (t_eff > total_duration) t_eff = total_duration;
                    }
                }

                // --- 2. Find Active Segment ---
                int seg = 0;
                while (seg < num_wp - 1 && t_eff >= wp_t[seg + 1]) {
                    seg++;
                }

                // --- 3. Interpolate Relative Offset ---
                float offset_x = 0.0f, offset_y = 0.0f, offset_z = 0.0f;
                
                if (seg < num_wp - 1) {
                    float seg_duration = wp_t[seg + 1] - wp_t[seg];
                    if (seg_duration > 0.0f) {
                        float factor = (t_eff - wp_t[seg]) / seg_duration;
                        offset_x = wp_x[seg] + factor * (wp_x[seg+1] - wp_x[seg]);
                        offset_y = wp_y[seg] + factor * (wp_y[seg+1] - wp_y[seg]);
                        offset_z = wp_z[seg] + factor * (wp_z[seg+1] - wp_z[seg]);
                    }
                } else {
                    // Time exceeded or exactly at final waypoint
                    offset_x = wp_x[num_wp - 1];
                    offset_y = wp_y[num_wp - 1];
                    offset_z = wp_z[num_wp - 1];
                }

                // --- 4. Apply Relative Offset to Base Coordinate ---
                source_xd_h[i] += offset_x;
                source_yd_h[i] += offset_y;
                source_zd_h[i] += offset_z;
            }
            fwp.close();
        }

		// Copy data from host to device //
		cudaMemcpy(source_xd_d, source_xd_h, num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(source_yd_d, source_yd_h, num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(source_zd_d, source_zd_h, num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(vel_us_d, 	vel_us_h,    num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(vel_vs_d, 	vel_vs_h,    num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(vel_ws_d, 	vel_ws_h,    num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(Group_d,     Group_h,     num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		cudaMemcpy(pos_idd_d,   pos_idd_h,   num_source*sizeof(FLOAT), cudaMemcpyHostToDevice);
		
		cudaThreadSynchronize(); // cudaThreadSynchronize //

		const int gen_step_ = pfrag.gen_step;

		// source //
		gpu_particle_source_LSM <<< grid, block >>> (
			ppos,
			index_num,
			parray_start,
			num   [0], num   [1], num   [2], 
			point [0], point [1], point [2],
			vec_dx[0], vec_dx[1], vec_dx[2],
			source_xd_d, source_yd_d, source_zd_d,
			vel_us_d,    vel_vs_d,    vel_ws_d,
			Group_d, pos_idd_d,
			t,
			pstart,
			gen_step_,
			c_ref		// MOD 2021
			);

		cudaThreadSynchronize(); // cudaThreadSynchronize //


		// region check //
		const FLOAT	x_range_min[3] = { domain.x_min, domain.y_min, domain.z_min };
		const FLOAT	x_range_max[3] = { domain.x_max, domain.y_max, domain.z_max };

		gpu_particle_region_check <<< grid, block >>> (
			ppos,
			parray_start,
			num[0], num[1], num[2], 
			x_range_min[0], x_range_min[1], x_range_min[2],
			x_range_max[0], x_range_max[1], x_range_max[2]);

		cudaThreadSynchronize(); // cudaThreadSynchronize //


		pgrid.parray_end += num_source;
		index_num++;

		cudaFree(source_xd_d);
		cudaFree(source_yd_d);
		cudaFree(source_zd_d);
		cudaFree(vel_us_d);
		cudaFree(vel_vs_d);
		cudaFree(vel_ws_d);
		cudaFree(Group_d);
		cudaFree(pos_idd_d);
	}
	else {
		std::cout << "Particle_Source_Box\n";
		std::cout << "number of particles is too large\n";
		exit(3);
	}
}

void 
particle_source_sphere (
	const Domain			&domain,
	      ParticleCalFlag	&pfrag, 
	      ParticleGrid		&pgrid,
	      ParticlePosition	*ppos,
	const int	num[],
	const FLOAT	point[],
	FLOAT		radius,
	const FLOAT	theta0[],
	const FLOAT	theta1[],
	int			&index_num)
{
	const int	num_source  = num[0] * num[1];

	const int	thread_length  = 256;
	const int	block_number   = num_source/thread_length + 1;


	dim3	grid,
			block;
	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length, 1, 1);


	int		parray_start = pgrid.parray_end;
	if (parray_start + num_source < pgrid.num_particle_max) {
		// kernel
		gpu_particle_source_sphere <<< grid, block >>> (
			ppos,
			index_num,
			parray_start,
			num   [0], num   [1],
			point [0], point [1], point [2],
			radius,
			theta0[0], theta0[1],
			theta1[0], theta1[1]
			);

		cudaThreadSynchronize(); // cudaThreadSynchronize //


		// region check //
		const FLOAT	x_range_min[3] = { domain.x_min, domain.y_min, domain.z_min };
		const FLOAT	x_range_max[3] = { domain.x_max, domain.y_max, domain.z_max };

		gpu_particle_region_check <<< grid, block >>> (
			ppos,
			parray_start,
			num[0], num[1], num[2], 
			x_range_min[0], x_range_min[1], x_range_min[2],
			x_range_max[0], x_range_max[1], x_range_max[2]);

		cudaThreadSynchronize(); // cudaThreadSynchronize //

		pgrid.parray_end += num_source;
		index_num++;
	}
	else {
		std::cout << "Particle_Source_Sphere\n";
		std::cout << "number of particles is too large\n";
		exit(3);
	}
}


// CalParticle_Generate.cu //
