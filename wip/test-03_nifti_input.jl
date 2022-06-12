# Initialize particles from a nifti file

include("/home/faruk/Git/thingsonthings/code/post016-julia_brainsplode/core.jl")

import Distributions
import ImageFiltering
import Images

OUTDIR = "/home/faruk/Documents/test_julia/test-03"
INPUT = "/home/faruk/Documents/test_julia/T1w_brain.nii.gz"

TARGET_FRAMERATE = 60  # FPS
DURATION = 12  # seconds
NR_ITERATIONS = TARGET_FRAMERATE * DURATION
TIME_STEP = 2

SMOOTH_FACTOR = 10
GRAVITY = -0.005

# =============================================================================
# Create a directory if it does not exist
if !isdir(OUTDIR)
    mkdir(OUTDIR)
end

# Load nifti
nii = NIfTI.niread(INPUT)

# Select a slice
data = nii.raw[:, 160, :]'
println(string("Slice dims: ", size(data), ", with type: ", typeof(data)))

# Resample image
data = Images.imresize(data, ratio=2);

# Normalize data to an arbitrary range
data .-= minimum(data)
data ./= maximum(data)
data .*= 150

# Save a PNG image
save_png(data, joinpath(OUTDIR, "input.png"), false)

# =============================================================================
# Initialize particles positions
p_idx = findall(data .> 0)
p_pos = Array{Float32}(getindex.(p_idx, [1 2])) .- 1

GRID_SIZE = size(data)
NR_PARTICLES = size(p_pos)[1]
println("Number of particles: ", NR_PARTICLES)
println("Grid size          : ", GRID_SIZE)

# Record voxel intensity values into particles
p_val = data[p_idx]

# Initialize velocities
p_vel = zeros((NR_PARTICLES, 2))
p_vel[:, 1] = rand(Distributions.Uniform(-1, 1), NR_PARTICLES)
p_vel[:, 2] = rand(Distributions.Uniform(-1, 1), NR_PARTICLES)

# =============================================================================
g_vel_x = zeros(GRID_SIZE)
g_vel_x = particle_to_grid_interpolate_quadratic(p_pos, g_vel_x, p_vel[:, 1])

g_vel_y = zeros(GRID_SIZE)
g_vel_y = particle_to_grid_interpolate_quadratic(p_pos, g_vel_y, p_vel[:, 2])

# Smooth scalar grids
g_vel_x = ImageFiltering.imfilter(
    g_vel_x, ImageFiltering.Kernel.gaussian(SMOOTH_FACTOR));
g_vel_y = ImageFiltering.imfilter(
    g_vel_y, ImageFiltering.Kernel.gaussian(SMOOTH_FACTOR));

# Grid to particle mapping
for i = 1:NR_PARTICLES
    x, y = p_pos[i, :]
    x = floor(Int, x)
    y = floor(Int, y)
    p_vel[i, 1] = g_vel_x[x, y]
    p_vel[i, 2] = g_vel_y[x, y]
end

# =============================================================================
# Move particles using their velocities
@time for i = 1:NR_ITERATIONS
    global p_pos, p_vel

    # Reset grid
    local grid = zeros(GRID_SIZE)

    # Map particle values to grid
    grid = particle_to_grid_interpolate_quadratic(p_pos, grid, p_val)
    save_png(grid, joinpath(OUTDIR, string("frame-", lpad(i, 4, "0"), ".png")),
             false)

    # Update particle positions
    p_pos += p_vel * TIME_STEP

    # Enforce boundary conditions
    p_pos, p_vel = clamp_and_bounce(p_pos, p_vel, GRID_SIZE[1], GRID_SIZE[2])

    # Add gradity
    p_vel[:, 2] .+= GRAVITY

end

println("Finished.")
