import os
import struct
import vtk
import paraview.simple as pvs

# ==========================================
# 1. SETTINGS
# ==========================================
cd_position = "./20260512_particle_cubes_waypoints_test" 
map_file    = "./map/map_cubes_small.dat" 
out_png_dir = "./20260512_PNG_ParaView_cubes_test"

x_grid    = 192
y_grid    = 96
z_grid    = 64
resolut   = 2.0
pout      = 100
dt        = 0.01

start_num = 0
data_num  = 249
dout      = 1
num_ranks = 1

if not os.path.exists(out_png_dir):
    os.makedirs(out_png_dir)

# ==========================================
# 2. READ AND BUILD THE TOPOGRAPHY MAP
# ==========================================
print("Reading topography map...")
z_data = []

with open(map_file, 'r') as f:
    header = f.readline() 
    for line in f:
        row = [float(val) for val in line.split()]
        if row:
            z_data.append(row)

sg = vtk.vtkStructuredGrid()
sg.SetDimensions(x_grid, y_grid, 1)
map_points = vtk.vtkPoints()

for j in range(y_grid):
    for i in range(x_grid):
        x = (i + 1) * resolut
        y = (j + 1) * resolut
        row_idx = (y_grid - 1) - j 
        z = z_data[row_idx][i]
        map_points.InsertNextPoint(x, y, z)

sg.SetPoints(map_points)

map_source = pvs.TrivialProducer()
map_source.GetClientSideObject().SetOutput(sg)
surface_filter = pvs.ExtractSurface(Input=map_source)

# ==========================================
# 3. SETUP THE PARAVIEW PIPELINE
# ==========================================
# FIX: Create the VTK containers EXACTLY ONCE outside the loop
vtk_points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)
polydata.SetVerts(vertices)

particle_source = pvs.TrivialProducer()
particle_source.GetClientSideObject().SetOutput(polydata)

# Setup View
view = pvs.CreateRenderView()
view.ViewSize = [1920, 1080]
view.Background = [0.0, 0.0, 0.0]

# Display Map
map_display = pvs.Show(surface_filter, view)
map_display.Representation = 'Surface'
map_display.AmbientColor = [0.6, 0.6, 0.6] 
map_display.DiffuseColor = [0.6, 0.6, 0.6]

time_text = pvs.Text()
time_display = pvs.Show(time_text, view)
time_display.WindowLocation = 'UpperCenter'
time_display.Color = [1.0, 1.0, 1.0]
time_display.FontSize = 14

# ==========================================
# 4. MAIN RENDER LOOP
# ==========================================
print("Starting ParaView Rendering Loop...")

is_first_frame = True

for i in range(start_num, data_num, dout):
    
    # FIX: Empty the existing containers instead of making new ones
    vtk_points.Reset()
    vertices.Initialize() 
    has_data = False
    
    for r in range(num_ranks):
        filepath = os.path.join(cd_position, "position{}-{}.bin".format(r, i))
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                
            num_floats = len(raw_data) // 4
            floats = struct.unpack('f' * num_floats, raw_data)
            floats = floats[1:] 
            
            num_particles = len(floats) // 3
            for p in range(num_particles):
                j = p * 3
                x = floats[j]
                y = (y_grid * resolut) - floats[j+1]
                z = floats[j+2]
                
                pid = vtk_points.InsertNextPoint(x, y, z)
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(pid)
                has_data = True
                
    if not has_data:
        print("Warning: No data for step {}".format(i))
        continue
        
    # FIX: Explicitly tell ParaView that the internal arrays have changed
    vtk_points.Modified()
    vertices.Modified()
    polydata.Modified()
    particle_source.UpdatePipeline()
    
    tt = i * pout * dt
    h = int(tt / 3600)
    m = int((tt % 3600) / 60)
    s = tt % 60
    time_text.Text = "3D View - Time {:02d}:{:02d}:{:02.0f}".format(h, m, s)
    
    # ---------------------------------------------------------
    # FIRST FRAME INITIALIZATION
    # ---------------------------------------------------------
    if is_first_frame:
        particle_display = pvs.Show(particle_source, view)
        particle_display.Representation = 'Points'
        particle_display.RenderPointsAsSpheres = 1
        particle_display.AmbientColor = [0.0, 1.0, 1.0]
        particle_display.PointSize = 6.0
        
        view.CameraViewUp = [0.0, 0.0, 1.0] 
        
        cx = (x_grid * resolut) / 2.0
        cy = (y_grid * resolut) / 2.0
        cz = 0.0
        view.CameraFocalPoint = [cx, cy, cz]
        view.CameraPosition = [cx * 2.5, cy * -1.5, (x_grid * resolut) * 0.8]
        
        pvs.ResetCamera(view)
        is_first_frame = False
    
    pvs.Render()
    
    out_file = os.path.join(out_png_dir, "frame-{:04d}.png".format(i))
    pvs.SaveScreenshot(out_file, view)
    
    if i % 10 == 0:
        print("Rendered step {}".format(i))

print("Visualization complete!")