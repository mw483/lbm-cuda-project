import numpy as np
import yaml
import os

map_config_dir = "map_config"
map_config_file = "tsubame_map_config.yaml"

full_path = os.path.join(".", map_config_dir, map_config_file)

class MapCanvas:
    """The core engine that holds the grid and handles exports."""
    def __init__(self, domain_x_m, domain_y_m, grid_res_m):
        self.grid_res_m = grid_res_m
        self.rows = int(domain_x_m / grid_res_m)
        self.cols = int(domain_y_m / grid_res_m)
        self.grid_array = np.zeros((self.rows, self.cols), dtype=int)
        print(f"Canvas initialized: {self.rows}x{self.cols} cells.")

    def add_rectangle(self, x_pos_m, y_pos_m, x_dim_m, y_dim_m, height, center_y=False):
        """Draws a rectangle onto the grid array."""
        rows_dim = int(x_dim_m / self.grid_res_m)
        cols_dim = int(y_dim_m / self.grid_res_m)
        
        row_start = int(x_pos_m / self.grid_res_m)
        row_end = row_start + rows_dim
        
        if center_y:
            col_start = (self.cols // 2) - (cols_dim // 2)
        else:
            col_start = int(y_pos_m / self.grid_res_m)
            
        col_end = col_start + cols_dim
        
        # Draw directly to the array
        self.grid_array[row_start:row_end, col_start:col_end] = height

    def add_staggered_cubes(self, x_start_m, x_end_m, y_start_m, y_end_m, cube_size_m, x_spacing_m, y_spacing_m, height):
        """
        Generates a staggered array of cubes within a defined bounding box.
        Guarantees perfect symmetry along the y-axis center of the bounding box.
        """
        print(f"Generating symmetrical staggered cubes from x={x_start_m}m to {x_end_m}m...")
        
        x_pos = x_start_m
        stagger_toggle = False
        count = 0
        
        # Calculate the Y-center of the bounding box
        box_center_y_m = (y_start_m + y_end_m) / 2.0
        available_y_space_m = y_end_m - y_start_m
        
        # Maximum normal cubes that can fit: N*C + (N-1)*S <= L  =>  N <= (L+S)/(C+S)
        max_normal_cubes = int((available_y_space_m + y_spacing_m) // (cube_size_m + y_spacing_m))
        
        while x_pos + cube_size_m <= x_end_m:
            
            if not stagger_toggle:
                # Normal row
                n_cubes = max_normal_cubes
            else:
                # Staggered row (use 1 less cube to create the stagger and keep it inside bounds)
                n_cubes = max_normal_cubes - 1
                if n_cubes < 1: n_cubes = 1 # Fallback for very narrow domains
            
            # Calculate the total physical length of this specific row
            row_length_m = (n_cubes * cube_size_m) + ((n_cubes - 1) * y_spacing_m)
            
            # Start position to perfectly center this row around the y-axis
            y_pos = box_center_y_m - (row_length_m / 2.0)
            
            # Draw the cubes for this row
            for _ in range(n_cubes):
                self.add_rectangle(x_pos, y_pos, cube_size_m, cube_size_m, height, center_y=False)
                y_pos += cube_size_m + y_spacing_m
                count += 1
            
            # Move to the next row in the x-direction
            x_pos += cube_size_m + x_spacing_m
            stagger_toggle = not stagger_toggle
            
        print(f"Successfully added {count} symmetrical staggered cubes.")

    def export_dat(self, filename):
        """Writes the LBM .dat file."""
        with open(filename, 'w') as f:
            f.write(f"{self.rows}\t{self.cols}\n")
            for row in self.grid_array.T:
                f.write("\t".join(map(str, row)) + "\n")
        print(f"Exported successfully to {filename}")


def generate_from_yaml(yaml_path):
    """The parser that reads the config and drives the Canvas."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 1. Setup Canvas
    dom = config['domain']
    canvas = MapCanvas(dom['x_m'], dom['y_m'], dom['res_m'])
    
    # 2. Add Obstacles
    for obs in config.get('obstacles', []):
        if obs['type'] == 'rectangle':
            canvas.add_rectangle(
                x_pos_m=obs['x_pos_m'],
                y_pos_m=obs.get('y_pos_m', 0),
                x_dim_m=obs['x_dim_m'],
                y_dim_m=obs['y_dim_m'],
                height=obs['height'],
                center_y=obs.get('center_y', False)
            )
            print(f"Parsed and added rectangle at x={obs['x_pos_m']}m.")
            
        elif obs['type'] == 'staggered_cubes':
            canvas.add_staggered_cubes(
                x_start_m=obs['x_start_m'],
                x_end_m=obs['x_end_m'],
                y_start_m=obs['y_start_m'],
                y_end_m=obs['y_end_m'],
                cube_size_m=obs['cube_size_m'],
                x_spacing_m=obs['x_spacing_m'],
                y_spacing_m=obs['y_spacing_m'],
                height=obs['height']
            )
            
    # 3. Export
    canvas.export_dat(config['output']['filename'])


if __name__ == "__main__":
    generate_from_yaml(full_path)