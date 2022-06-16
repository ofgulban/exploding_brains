# Initialize particles from a nifti file
# Reference: https://nialltl.neocities.org/articles/mpm_guide.html


include("/home/faruk/Git/thingsonthings/code/post016-julia_brainsplode/core.jl")

import Distributions
import ImageFiltering
import Images

OUTDIR = "/home/faruk/Documents/test_julia/test-05"
INPUT = "/home/faruk/Documents/test_julia/T1w_brain.nii.gz"

TARGET_FRAMERATE = 60  # FPS
DURATION = 3  # seconds
NR_ITERATIONS = TARGET_FRAMERATE * DURATION
TIME_STEP = 0.1

SMOOTH_FACTOR = 10
GRAVITY = -0.3

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
println("Number of frames   : ", NR_ITERATIONS)

# Record voxel intensity values into particles
p_val = data[p_idx]
# Initialize particle masses
p_mass = ones(NR_PARTICLES)
# Initialize affine momentum matrix
p_C = zeros((NR_PARTICLES, 2, 2))
# Initialize deformation gradient to identity matrix
p_Fs = zeros((NR_PARTICLES, 2, 2))
p_Fs[:, 1, 1] .= 1
p_Fs[:, 2, 2] .= 1

# Initialize velocities
p_vel = zeros((NR_PARTICLES, 2))
p_vel[:, 1] = rand(Distributions.Uniform(-1, 1), NR_PARTICLES)
p_vel[:, 2] = rand(Distributions.Uniform(-1, 1), NR_PARTICLES)

p_vel .*= 10

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
    global p_pos, p_vel, p_vol
    local grid, g_mass

    # Compute particle volumes
    grid = zeros(GRID_SIZE)
    g_mass = particle_to_grid_interpolate_quadratic(p_pos, grid, p_mass)
    p_vol = grid_to_particle_estimate_volume(p_pos, g_mass, p_mass)

    # Map particle values to grid
    grid = zeros(GRID_SIZE)
    grid = particle_to_grid_interpolate_quadratic(p_pos, grid, p_val)
    save_png(grid, joinpath(OUTDIR, string("frame-", lpad(i, 4, "0"), ".png")),
    false)

    # Compute velocity grid
    local g_vel = particle_to_grid_velocity(p_pos, grid, p_vel, p_mass, p_C, p_Fs, p_vol)

    # Grid to particle mapping
    Threads.@threads for i = 1:NR_PARTICLES
        # Get a particle's position
        p_x, p_y = p_pos[i, :]

        # Compute which grid cell particle falls within
        g_x::Int = floor(p_x)
        g_y::Int = floor(p_y)

        # Convert grid cell index to cell center coordinate
        cell_center_x = g_x + 0.5
        cell_center_y = g_y + 0.5

        # Compute difference between grid cell index and particle position
        p_diff_x = p_x - cell_center_x
        p_diff_y = p_y - cell_center_y

        # Compute quadratic weights
        wx_i = 0.5 * (0.5 - p_diff_x)^2
        wx_j = 0.75 - p_diff_x^2
        wx_k = 0.5 * (0.5 + p_diff_x)^2

        wy_i = 0.5 * (0.5 - p_diff_y)^2
        wy_j = 0.75 - p_diff_y^2
        wy_k = 0.5 * (0.5 + p_diff_y)^2

        p_vel[i, 1] = 0
        p_vel[i, 1] += wx_i * wy_i * g_vel[g_x-1, g_y-1, 1]
        p_vel[i, 1] += wx_j * wy_i * g_vel[g_x  , g_y-1, 1]
        p_vel[i, 1] += wx_k * wy_i * g_vel[g_x+1, g_y-1, 1]
        p_vel[i, 1] += wx_i * wy_j * g_vel[g_x-1, g_y  , 1]
        p_vel[i, 1] += wx_j * wy_j * g_vel[g_x  , g_y  , 1]
        p_vel[i, 1] += wx_k * wy_j * g_vel[g_x+1, g_y  , 1]
        p_vel[i, 1] += wx_i * wy_k * g_vel[g_x-1, g_y+1, 1]
        p_vel[i, 1] += wx_j * wy_k * g_vel[g_x  , g_y+1, 1]
        p_vel[i, 1] += wx_k * wy_k * g_vel[g_x+1, g_y+1, 1]

        p_vel[i, 2] = 0
        p_vel[i, 2] += wx_i * wy_i * g_vel[g_x-1, g_y-1, 2]
        p_vel[i, 2] += wx_j * wy_i * g_vel[g_x  , g_y-1, 2]
        p_vel[i, 2] += wx_k * wy_i * g_vel[g_x+1, g_y-1, 2]
        p_vel[i, 2] += wx_i * wy_j * g_vel[g_x-1, g_y  , 2]
        p_vel[i, 2] += wx_j * wy_j * g_vel[g_x  , g_y  , 2]
        p_vel[i, 2] += wx_k * wy_j * g_vel[g_x+1, g_y  , 2]
        p_vel[i, 2] += wx_i * wy_k * g_vel[g_x-1, g_y+1, 2]
        p_vel[i, 2] += wx_j * wy_k * g_vel[g_x  , g_y+1, 2]
        p_vel[i, 2] += wx_k * wy_k * g_vel[g_x+1, g_y+1, 2]

        # ---------------------------------------------------------------------
        # NOTE[Faruk]: I should double check this section.
        # Compute cell distances to particle
        cell_dist_x_i = p_x - (cell_center_x - 1)
        cell_dist_x_j = p_x - cell_center_x
        cell_dist_x_k = p_x - (cell_center_x + 1)

        cell_dist_y_i = p_y - (cell_center_y - 1)
        cell_dist_y_j = p_y - cell_center_y
        cell_dist_y_k = p_y - (cell_center_y + 1)

        # Constructing affine per-particle momentum matrix from APIC / MLS-MPM.
        # See APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
        # elow equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
        # where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
        B = zeros((2, 2))
        B[1, 1] += wx_i * wy_i * g_vel[g_x-1, g_y-1, 1] * cell_dist_x_i
        B[1, 2] += wx_i * wy_i * g_vel[g_x-1, g_y-1, 2] * cell_dist_x_i
        B[2, 1] += wx_i * wy_i * g_vel[g_x-1, g_y-1, 1] * cell_dist_y_i
        B[2, 2] += wx_i * wy_i * g_vel[g_x-1, g_y-1, 2] * cell_dist_y_i

        B[1, 1] += wx_j * wy_i * g_vel[g_x  , g_y-1, 1] * cell_dist_x_j
        B[1, 2] += wx_j * wy_i * g_vel[g_x  , g_y-1, 2] * cell_dist_x_j
        B[2, 1] += wx_j * wy_i * g_vel[g_x  , g_y-1, 1] * cell_dist_y_i
        B[2, 2] += wx_j * wy_i * g_vel[g_x  , g_y-1, 2] * cell_dist_y_i

        B[1, 1] += wx_k * wy_i * g_vel[g_x+1, g_y-1, 1] * cell_dist_x_k
        B[1, 2] += wx_k * wy_i * g_vel[g_x+1, g_y-1, 2] * cell_dist_x_k
        B[2, 1] += wx_k * wy_i * g_vel[g_x+1, g_y-1, 1] * cell_dist_y_i
        B[2, 2] += wx_k * wy_i * g_vel[g_x+1, g_y-1, 2] * cell_dist_y_i

        B[1, 1] += wx_i * wy_j * g_vel[g_x-1, g_y, 1] * cell_dist_x_i
        B[1, 2] += wx_i * wy_j * g_vel[g_x-1, g_y, 2] * cell_dist_x_i
        B[2, 1] += wx_i * wy_j * g_vel[g_x-1, g_y, 1] * cell_dist_y_j
        B[2, 2] += wx_i * wy_j * g_vel[g_x-1, g_y, 2] * cell_dist_y_j

        B[1, 1] += wx_j * wy_j * g_vel[g_x  , g_y, 1] * cell_dist_x_j
        B[1, 2] += wx_j * wy_j * g_vel[g_x  , g_y, 2] * cell_dist_x_j
        B[2, 1] += wx_j * wy_j * g_vel[g_x  , g_y, 1] * cell_dist_y_j
        B[2, 2] += wx_j * wy_j * g_vel[g_x  , g_y, 2] * cell_dist_y_j

        B[1, 1] += wx_k * wy_j * g_vel[g_x+1, g_y, 1] * cell_dist_x_k
        B[1, 2] += wx_k * wy_j * g_vel[g_x+1, g_y, 2] * cell_dist_x_k
        B[2, 1] += wx_k * wy_j * g_vel[g_x+1, g_y, 1] * cell_dist_y_j
        B[2, 2] += wx_k * wy_j * g_vel[g_x+1, g_y, 2] * cell_dist_y_j

        B[1, 1] += wx_i * wy_k * g_vel[g_x-1, g_y+1, 1] * cell_dist_x_i
        B[1, 2] += wx_i * wy_k * g_vel[g_x-1, g_y+1, 2] * cell_dist_x_i
        B[2, 1] += wx_i * wy_k * g_vel[g_x-1, g_y+1, 1] * cell_dist_y_k
        B[2, 2] += wx_i * wy_k * g_vel[g_x-1, g_y+1, 2] * cell_dist_y_k

        B[1, 1] += wx_j * wy_k * g_vel[g_x  , g_y+1, 1] * cell_dist_x_j
        B[1, 2] += wx_j * wy_k * g_vel[g_x  , g_y+1, 2] * cell_dist_x_j
        B[2, 1] += wx_j * wy_k * g_vel[g_x  , g_y+1, 1] * cell_dist_y_k
        B[2, 2] += wx_j * wy_k * g_vel[g_x  , g_y+1, 2] * cell_dist_y_k

        B[1, 1] += wx_k * wy_k * g_vel[g_x+1, g_y+1, 1] * cell_dist_x_k
        B[1, 2] += wx_k * wy_k * g_vel[g_x+1, g_y+1, 2] * cell_dist_x_k
        B[2, 1] += wx_k * wy_k * g_vel[g_x+1, g_y+1, 1] * cell_dist_y_k
        B[2, 2] += wx_k * wy_k * g_vel[g_x+1, g_y+1, 2] * cell_dist_y_k

        p_C[i, :, :] = B .* 4

        # Deformation gradient update - MPM course, equation 181
        # Fp' = (I + dt * p.C) * Fp
        # Fp_new = [[1., 0.] [0., 1.]]
        # Fp_new += TIME_STEP .* p_C[i, :, :]
        # p_Fs[i, :, :] = Fp_new * p_Fs[i, :, :]
    end

    # Add gradity
    p_vel[:, 2] .+= GRAVITY * TIME_STEP

    # Update particle positions
    p_pos += p_vel * TIME_STEP

    # Enforce boundary conditions
    p_pos, p_vel = clamp_and_bounce(p_pos, p_vel, GRID_SIZE[1], GRID_SIZE[2])

end

println("Finished.")
