import os
import struct
import paraview.simple as pvs
from vtk import vtkPoints, vtkPolyData

# ==========================================
# 1. SETTINGS (Match your MATLAB script)
# ==========================================
cd_position = "20260512_particle_cubes_waypoints_test" # Directory with the .bin files
out_png_dir = "20260512_PNG_ParaView_cubes_test"       # Output folder for PNGs

# Updated Grid Settings
x_grid    = 192
y_grid    = 96
z_grid    = 64
resolut   = 2.0
pout      = 100
dt        = 0.01

# Updated Loop Settings
start_num = 0
data_num  = 249
dout      = 1
num_ranks = 1

# Create output directory
if not os.path.exists(out_png_dir):
    os.makedirs(out_png_dir)

# ==========================================
# 2. SETUP THE PARAVIEW PIPELINE
# ==========================================
# We create empty VTK objects to hold our data. 
# We will update these inside the loop later.
vtk_points = vtkPoints()
polydata = vtkPolyData()
polydata.SetPoints(vtk_points)

# Wrap the VTK data into a ParaView pipeline object
particle_source = pvs.TrivialProducer()
particle_source.GetClientSideObject().SetOutput(polydata)

# Set up the View (The "Canvas")
view = pvs.CreateRenderView()
view.ViewSize = [1920, 1080] # 1080p Resolution
view.Background = [0.0, 0.0, 0.0] # Black background

# Display the particles in the view
particle_display = pvs.Show(particle_source, view)
particle_display.Representation = 'Points'
particle_display.AmbientColor = [0.0, 1.0, 1.0] # Cyan color [R, G, B]
particle_display.PointSize = 3.0 # Size of the dots

# Set the Camera (Mimics MATLAB's view(35, 50))
# These coordinates might need tweaking depending on your exact domain size
view.CameraPosition = [x_grid*resolut*1.5, y_grid*resolut*2.0, z_grid*resolut*1.5]
view.CameraFocalPoint = [x_grid*resolut/2.0, y_grid*resolut/2.0, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0] # Z is up

# Set up a Text Annotation for the Time
time_text = pvs.Text()
time_display = pvs.Show(time_text, view)
time_display.WindowLocation = 'UpperCenter'
time_display.Color = [1.0, 1.0, 1.0] # White text
time_display.FontSize = 14

# ==========================================
# 3. MAIN RENDER LOOP
# ==========================================
print("Starting ParaView Rendering Loop...")

for i in range(start_num, data_num, dout):
    # Clear the old points
    vtk_points.Reset()
    has_data = False
    
    # Read data from all ranks
    for r in range(num_ranks):
        filepath = os.path.join(cd_position, "position{}-{}.bin".format(r, i))
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                
            # Unpack the binary floats (MATLAB fread equivalent)
            num_floats = len(raw_data) // 4
            floats = struct.unpack('f' * num_floats, raw_data)
            
            # Strip Fortran record markers (first and last elements)
            # and loop through by 3s (X, Y, Z)
            floats = floats[1:-1]
            for j in range(0, len(floats), 3):
                x = floats[j]
                y = (y_grid * resolut) - floats[j+1] # Y-inversion from MATLAB
                z = floats[j+2]
                vtk_points.InsertNextPoint(x, y, z)
                has_data = True
                
    if not has_data:
        print("Warning: No data for step {}".format(i))
        continue
        
    # Tell ParaView the data has changed
    vtk_points.Modified()
    polydata.Modified()
    particle_source.UpdatePipeline()
    
    # Calculate and update time text
    tt = i * pout * dt
    h = int(tt / 3600)
    m = int((tt % 3600) / 60)
    s = tt % 60
    time_text.Text = "3D View - Time {:02d}:{:02d}:{:02.0f}".format(h, m, s)
    
    # Render and Save
    out_file = os.path.join(out_png_dir, "frame-{:04d}.png".format(i))
    pvs.SaveScreenshot(out_file, view)
    
    if i % 20 == 0:
        print("Rendered step {}".format(i))

print("Visualization complete!")