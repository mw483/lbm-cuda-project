import config
import os
import numpy as np

def get_map_dimensions(map_path):
    if not os.path.exists(map_path):
        print(f"Error, {map_path} not found")
        return None
    
    with open(map_path, 'r') as f:
        # Read first line of the map file, split by tab or whitespace
        line = f.readline().strip().split()
        nx, ny = [int(line[0]), int(line[1])]
    return nx, ny

def generate_sh(params, nx, ny):
    p = params['runlbm.sh']
    map = params['map']

    domain_x_m = nx * map['physical_dx']
    domain_y_m = ny * map['physical_dx']
    # Construct the multi-line command string
    # We use a f-string with backslashes for the shell script formatting
    cnn_z = int(p['length_z'] / 2)
    content = f"""#!/bin/sh
rm run
ln -s ./src/run
export OMP_NUM_THREADS=1
echo "gpu_linux slots=1" > hostfile.txt

mpirun -hostfile hostfile.txt -np 1 ./run \\
        -Time                           {p['Time']} \\
        -time_coef                      {p['time_coef']} \\
        -NMPI                           1       1       1 \\
        -CNN                            {cnn_z}              2 \\
        -velocity_lbm                   {p['velocity_lbm']}     0.02 \\
        -gpu_per_node                   1 \\
        -halo_grid                      1 \\
        -CFout                          500     60000 \\
        -CFRfrg                         1       0       0 \\
        -restart                        0 \\
        -fstart                         0 \\
        -domain_min                     -0.08   -0.08   -0.08 \\
        -length                         {domain_x_m}     {domain_y_m}     {p['length_z']} \\
        -ncpu_div                       1       1       1       1 \\
        -flag_particle_generate         {p['flag_particle_generate']} \\
        -prestart                       0 \\
        -pout                           {p['pout']} \\
        -pstartstep                     0 \\
        -particle                       {p['max_particles']} \\
        -generate_step                  {p['generate_step']} \\
        | tee  -a  log_t2sub.txt
"""
    
    with open("runlbm.sh", "w", newline='\n') as f:
        f.write(content)
    
    # Make the script executable automatically
    os.chmod("runlbm.sh", 0o755)
    print(f"Successfully generated runlbm.sh for domain: {domain_x_m}x{domain_y_m}x{p['length_z']} and map: {map['path']}")

def generate_define_user(params):
    p_h = params['Define_user.h']
    p_init = p_h['init']
    p_out = p_h['output']
    p_flags = p_h['flags']

    # 1. Format the arrays for C++ syntax: [1, 2] -> "{1, 2}"
    kout_str = "{" + ", ".join(map(str, p_out['kout'])) + "}"
    jout_str = "{" + ", ".join(map(str, p_out['jout'])) + "}"
    iout_str = "{" + ", ".join(map(str, p_out['iout'])) + "}"

    # 2. Read the template
    template_path = "./src/Define_user.h.template"
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return

    with open(template_path, 'r') as f:
        content = f.read()

    # 3. Perform safe text replacements
    replacements = {
        "{{DTDZ_LOW}}": str(p_init['DTDZ_LOW']),
        "{{DTDZ_HIGH}}": str(p_init['DTDZ_HIGH']),
        "{{HF}}": str(p_init['hf']),
        
        "{{AVG_INT}}": str(p_out['average_interval']),
        "{{SKIP_TIME}}": str(p_out['skip_time']),
        "{{OUT_INT_INS}}": str(p_out['output_interval_ins']),
        "{{TIME_OUT_INI}}": str(p_out['time_output_ins_ini']),
        
        # Auto-calculate array lengths to prevent kernel crashes
        "{{NZ_OUT}}": str(len(p_out['kout'])),
        "{{NJ_OUT}}": str(len(p_out['jout'])),
        "{{NI_OUT}}": str(len(p_out['iout'])),
        
        "{{KOUT_ARRAY}}": kout_str,
        "{{JOUT_ARRAY}}": jout_str,
        "{{IOUT_ARRAY}}": iout_str,
        
        "{{FLG_BUOY}}": str(p_flags['flg_buoyancy']),
        "{{FLG_SCALAR}}": str(p_flags['flg_scalar']),
        "{{FLG_PARTICLE}}": str(p_flags['flg_particle'])
    }

    for key, val in replacements.items():
        content = content.replace(key, val)

    # 4. Write the final C++ header file
    output_path = "./src/Define_user.h"
    with open(output_path, 'w') as f:
        f.write(content)
        
    print(f"Define_user.h generated successfully. (nz_out auto-set to {len(p_out['kout'])})")

def generate_param_fluid(params):
    # Extract the map path directly from the JSON
    m_path = params['map']['path']
    
    template_path = "./src/paramFluidProperty.cu.template"
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return

    # Read the template
    with open(template_path, 'r') as f:
        content = f.read()

    # Inject the map path
    content = content.replace("{{MAP_PATH}}", m_path)

    # Write the active CUDA source file
    output_path = "./src/paramFluidProperty.cu"
    with open(output_path, 'w') as f:
        f.write(content)
        
    print(f"paramFluidProperty.cu generated successfully. (Map path set to: {m_path})")

class ParticleGenerator:
    """
    Generates particle source files for LBM simulation.
    Features:
    - Uniform grid generation (Ignores obstacles/blocks).
    - Configurable generation area (X and Y limits).
    - Custom ID formatting (Source_Index * 10000 + 1).
    """
    def __init__(self, domain_x_m, domain_y_m, output_dir):
        self.Lx = domain_x_m
        self.Ly = domain_y_m
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_sources(self, spacing_x, spacing_y, heights, 
                         velocity=(0.0, 0.0, 0.13), group_number=1,
                         x_max_m=None, y_min_m=0.0, y_max_m=None,
                         filename_pos="particle_position_cubes_small_test.txt", filename_num="particle_number_cubes_small_test.txt"):
        """
        Generates particles in a uniform grid within specified limits.
        
        Args:
            spacing_x, spacing_y (float): Spacing in meters.
            heights (list): Z-levels.
            x_max_m (float): Max X limit. Defaults to Lx / 1.333 if None.
            y_min_m (float): Min Y limit. Defaults to 0.
            y_max_m (float): Max Y limit. Defaults to Ly if None.
        """
        
        # --- Default Limits ---
        if x_max_m is None:
            x_max_m = self.Lx / 1.333
        if y_max_m is None:
            y_max_m = self.Ly

        print(f"--- Generating Uniform Particles ---")
        print(f"Spacing: dx={spacing_x}m, dy={spacing_y}m")
        print(f"Heights: {heights}")
        print(f"Area Limits: X[0.0 to {x_max_m:.2f}], Y[{y_min_m:.2f} to {y_max_m:.2f}]")
        
        particles = []
        u, v, w = velocity
        
        # --- 1. Define Grid ---
        # Strategy: Generate grid for the whole requested max area, then filter.
        # This keeps the grid aligned to 0.0 regardless of where you start clipping y.
        
        # Generate raw coords (anchored to 0.0 + half spacing)
        # We generate up to x_max_m
        raw_x = np.arange(spacing_x/2, x_max_m, spacing_x)
        
        # We generate up to y_max_m (full width generation then trim is safer for alignment)
        raw_y = np.arange(spacing_y/2, self.Ly, spacing_y)
        
        # Filter Y coordinates to be within [y_min_m, y_max_m]
        # This preserves the grid alignment (e.g. 2.0, 6.0, 10.0...) even if you start at y=5.0
        y_coords = raw_y[(raw_y >= y_min_m) & (raw_y <= y_max_m)]
        x_coords = raw_x # Already limited by arange stop
        
        print(f"Grid Size: {len(x_coords)} columns x {len(y_coords)} rows x {len(heights)} heights")

        # --- 2. Iterate and Generate ---
        source_index = 1
        
        for x in x_coords:
            for y in y_coords:
                for z in heights:
                    
                    # Source 1 -> 10001, Source 2 -> 20001
                    current_id = (source_index * 10000) + 1
                    
                    # Add Particle
                    p_data = (x, y, z, u, v, w, group_number, current_id)
                    particles.append(p_data)
                    
                    source_index += 1

        # --- Summary ---
        print(f"Generated: {len(particles)} sources")
        print(f"Last ID: {current_id}")
        
        # --- Write Files ---
        filepath_pos = os.path.join(self.output_dir, filename_pos)
        print(f"Writing positions to: {filepath_pos}")
        with open(filepath_pos, 'w') as f:
            for p in particles:
                # Format: x y z u v w group id
                line = f"{p[0]:.6f}\t{p[1]:.6f}\t{p[2]:.6f}\t{p[3]:.6f}\t{p[4]:.6f}\t{p[5]:.6f}\t{p[6]}\t{p[7]}\n"
                f.write(line)
                
        filepath_num = os.path.join(self.output_dir, filename_num)
        print(f"Writing count to: {filepath_num}")
        with open(filepath_num, 'w') as f:
            f.write(str(len(particles)))
            
        print("Done.")

def generate_particles(params, nx, ny):
    p_map = params['map']
    domain_x_m = nx * p_map['physical_dx']
    domain_y_m = ny * p_map['physical_dx']
    output_dir = "./particle_position"

    gen = ParticleGenerator(domain_x_m, domain_y_m, output_dir)
    
    all_particles = []
    all_waypoints = []
    source_index = 1 # Master tracker for unique IDs

    # ADD THIS LINE HERE:
    MODE_MAP = {"once": 0, "pingpong": 1, "loop": 2}
    source_index = 1 # Master tracker for unique IDs

    # Loop through every source in your config
    for source in config.PARTICLE_SOURCES:
        
        if source["type"] == "uniform":
            x_limit = domain_x_m / source['x_max_ratio']
            y_min = source['y_padding']
            y_max = domain_y_m - source['y_padding']
            
            raw_x = np.arange(source['spacing_x']/2, x_limit, source['spacing_x'])
            raw_y = np.arange(source['spacing_y']/2, domain_y_m, source['spacing_y'])
            y_coords = raw_y[(raw_y >= y_min) & (raw_y <= y_max)]
            
            u, v, w = source['velocity']
            
            for x in raw_x:
                for y in y_coords:
                    for z in source['heights']:
                        current_id = (source_index * 10000) + 1
                        all_particles.append((x, y, z, u, v, w, source['group'], current_id))
                        
                        # NEW: Add a dummy waypoint for this static particle
                        # Format: Mode(0) NumWp(1) Time(0.0) dX(0.0) dY(0.0) dZ(0.0)
                        all_waypoints.append("0\t1\t0.000000\t0.000000\t0.000000\t0.000000")
                        source_index += 1
                        
        elif source["type"] == "point":
            # Native Python handling for a direct point source
            x, y, z = source["coords"]
            u, v, w = source["velocity"]
            current_id = (source_index * 10000) + 1
            all_particles.append((x, y, z, u, v, w, source["group"], current_id))
            
            # NEW: Add a dummy waypoint for this static particle
            all_waypoints.append("0\t1\t0.000000\t0.000000\t0.000000\t0.000000")
            source_index += 1
            
        elif source["type"] == "line":
            # Future placeholder for line logic
            pass

        elif source["type"] == "waypoint":
            u, v, w = source["velocity"]
            mode_int = MODE_MAP.get(source.get("mode", "once"), 0)
            
            # --- Logic: Determine Emitter Geometry ---
            emitter_coords = []
            if source.get("geometry") == "line":
                s = np.array(source["start_pos"])
                e = np.array(source["end_pos"])
                n_pts = source["num_points"]
                # Interpolate 5 points along the line
                for i in range(n_pts):
                    # Linear interpolation: P = Start + (i/(n-1))*(End - Start)
                    frac = i / (n_pts - 1) if n_pts > 1 else 0
                    p = s + frac * (e - s)
                    emitter_coords.append(p)
            else:
                # Default to single point
                emitter_coords.append(np.array(source["coords"]))

            # --- Logic: Generate the 10x Stack ---
            per_point = source.get("particles_per_point", 1)
            for coord in emitter_coords:
                for _ in range(per_point):
                    current_id = (source_index * 10000) + 1
                    # Add to master position list
                    all_particles.append((coord[0], coord[1], coord[2], u, v, w, source['group'], current_id))
                    
                    # Store waypoint info for this specific ID
                    wps = source["waypoints"]
                    wp_line = f"{mode_int}\t{len(wps)}\t" + "\t".join([f"{val:.6f}" for wp in wps for val in wp])
                    all_waypoints.append(wp_line)
                    
                    source_index += 1

    # --- Write Files ---
    os.makedirs(output_dir, exist_ok=True)
    out_settings = config.PARTICLE_OUTPUT
    
    filepath_pos = os.path.join(output_dir, out_settings['filename_pos'])
    print(f"Writing {len(all_particles)} total positions to: {filepath_pos}")
    with open(filepath_pos, 'w', newline='\n') as f:
        for p in all_particles:
            line = f"{p[0]:.6f}\t{p[1]:.6f}\t{p[2]:.6f}\t{p[3]:.6f}\t{p[4]:.6f}\t{p[5]:.6f}\t{p[6]}\t{p[7]}\n"
            f.write(line)
            
    filepath_num = os.path.join(output_dir, out_settings['filename_num'])
    with open(filepath_num, 'w', newline='\n') as f:
        f.write(str(len(all_particles)))
    
    filepath_wp = os.path.join(output_dir, "particle_waypoints.txt")
    with open(filepath_wp, 'w', newline='\n') as f:
        for wp_row in all_waypoints:
            f.write(wp_row + "\n")

def generate_read_particle_box(params):
    p_read = params['read_particle_box']
    output_dir = "./read_particle_box"
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the numbers exactly as the C++ stream expects them
    content = f"{p_read['pstart']} \t {p_read['pnum']}\n"
    content += f"{p_read['num_g'][0]} \t {p_read['num_g'][1]} \t {p_read['num_g'][2]}\n"
    content += f"{p_read['point_g'][0]:.6f} \t {p_read['point_g'][1]:.6f} \t {p_read['point_g'][2]:.6f}\n"
    content += f"{p_read['vec_g'][0]:.6f} \t {p_read['vec_g'][1]:.6f} \t {p_read['vec_g'][2]:.6f}\n"
    
    with open(os.path.join(output_dir, "read_particle_box.txt"), 'w') as f:
        f.write(content)
        
    print("read_particle_box.txt generated successfully.")

if __name__ == "__main__":
    data = config.PARAMS

    m_path = data['map']['path']
    dims = get_map_dimensions(m_path)

    if dims:
        nx, ny = dims
        generate_sh(data, nx, ny)
        generate_define_user(data)
        generate_param_fluid(data)
        generate_particles(data, nx, ny)
        generate_read_particle_box(data)
