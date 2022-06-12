# Move particles

include("/home/faruk/Git/thingsonthings/code/post016-julia_brainsplode/core.jl")

import Distributions

OUTDIR = "/home/faruk/Documents/test_julia/test-01"

GRID_SIZE = (512, 512)
NR_PARTICLES = 1024

TARGET_FRAMERATE = 60  # FPS
DURATION = 6  # seconds
NR_ITERATIONS = TARGET_FRAMERATE * DURATION
TIME_STEP = 0.5


# =============================================================================
# Create a directory if it does not exist
if !isdir(OUTDIR)
    mkdir(OUTDIR)
end

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

    # Clamp particles within bounds
    p_pos, p_vel = clamp_and_bounce(p_pos, p_vel, GRID_SIZE[1], GRID_SIZE[2])

end

println("Finished.")
