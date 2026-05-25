import pandas as pd
import numpy as np
import pickle

# Load the file
with open('flat_bld_m40_40_bld30.pkl', 'rb') as file:
    df = pickle.load(file)

# Convert to numpy for easier masking
grid = df.to_numpy()

# Find all coordinates where the height is greater than 0
y_coords, x_coords = np.where(grid > 0)

if len(y_coords) > 0:
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    max_height = grid.max()
    
    print("--- Building Details Extracted ---")
    print(f"Width (X cells): {max_x - min_x + 1}")
    print(f"Length (Y cells): {max_y - min_y + 1}")
    print(f"Max Height: {max_height}")
else:
    print("The grid is completely empty (all zeros).")