# Move particles with a radial velocity field

include("/home/faruk/Git/thingsonthings/code/post016-julia_brainsplode/core.jl")

import Distributions
import ImageFiltering

OUTDIR = "/home/faruk/Documents/test_julia/test-04"

GRID_SIZE = (512, 512)
NR_PARTICLES = 2^16

TARGET_FRAMERATE = 60  # FPS
DURATION = 12  # seconds
NR_ITERATIONS = TARGET_FRAMERATE * DURATION

TIME_STEP = 2
SMOOTH_FACTOR = 7.5
GRAVITY = -0.005

# =============================================================================
# Create a directory if it does not exist
if !isdir(OUTDIR)
    mkdir(OUTDIR)
end

println("Number of particles: ", NR_PARTICLES)

# =============================================================================
# Initialize particle positions
p_pos = zeros((NR_PARTICLES, 2))
p_pos[:, 1] = rand(Distributions.Uniform(2, GRID_SIZE[1]), NR_PARTICLES)
p_pos[:, 2] = rand(Distributions.Uniform(2, GRID_SIZE[2]), NR_PARTICLES)

# Initialize particle values
p_val = ones(NR_PARTICLES) .* 150

# Initialize velocities
p_vel = zeros((NR_PARTICLES, 2))
p_vel[:, 1] = rand(Distributions.Uniform(-1, 1), NR_PARTICLES)
p_vel[:, 2] = rand(Distributions.Uniform(-1, 1), NR_PARTICLES)

# -----------------------------------------------------------------------------
# Generate a radial velocity field
g_velo = zeros(Float32, (GRID_SIZE[1], GRID_SIZE[2], 2))
g_norm = zeros(Float32, GRID_SIZE)
vector = Array{Float32}(undef, 2)
for x = 1:GRID_SIZE[1]
    for y = 1:GRID_SIZE[2]
        # Compute x and y elements, adjust origin in case of even numbers
        if GRID_SIZE[1] % 2 == 0
            vector[1] = x - GRID_SIZE[1]/2 + 0.5
        else
            vector[1] = x - GRID_SIZE[1]/2
        end

        if GRID_SIZE[2] % 2 == 0
            vector[2] = y - GRID_SIZE[2]/2 + 0.5
        else
            vector[1] = x - GRID_SIZE[1]/2
        end

        # Compute L2 norm
        norm = LinearAlgebra.norm(vector)

        # Convert to polar
        theta = atan(vector[1], vector[2])

        # Normalize with norm
        if norm > 0
            vector ./= norm
        end

        # Assign values to image cells
        g_norm[x, y] = norm
        g_velo[x, y, :] = vector
    end
end

# Grid to particle mapping
for i = 1:NR_PARTICLES
    x, y = p_pos[i, :]
    x = floor(Int, x)
    y = floor(Int, y)
    p_vel[i, :] = g_velo[x, y, :]
end

# =============================================================================
g_vel_x = zeros(GRID_SIZE)
g_vel_x = particle_to_grid_interpolate_quadratic(p_pos, g_vel_x, p_vel[:, 1])

g_vel_y = zeros(GRID_SIZE)
g_vel_y = particle_to_grid_interpolate_quadratic(p_pos, g_vel_y, p_vel[:, 2])

# Smooth scalar grids
g_vel_x = ImageFiltering.imfilter(
    g_vel_x, ImageFiltering.Kernel.gaussian(SMOOTH_FACTOR))
g_vel_y = ImageFiltering.imfilter(
    g_vel_y, ImageFiltering.Kernel.gaussian(SMOOTH_FACTOR))

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
    grid = particle_to_grid_interpolate_quadratic(p_pos, grid, p_val)
    save_png(grid, joinpath(OUTDIR, string("frame-", lpad(i, 4, "0"), ".png")),
             false)

    # Update particle positions
    p_pos += p_vel * TIME_STEP

    # Enforce boundary conditions
    p_pos, p_vel = clamp_and_bounce(p_pos, p_vel, GRID_SIZE[1], GRID_SIZE[2])

    # Add gravity
    p_vel[:, 2] .+= GRAVITY

end

println("Finished.")
