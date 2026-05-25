import numpy as np
import yaml

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
        
        # Draw directly to the array (bounds checking omitted for brevity)
        self.grid_array[row_start:row_end, col_start:col_end] = height
        print(f"Added building: {rows_dim}x{cols_dim} cells at height {height}.")

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
            
    # 3. Export
    canvas.export_dat(config['output']['filename'])

if __name__ == "__main__":
    generate_from_yaml("map_config.yaml")