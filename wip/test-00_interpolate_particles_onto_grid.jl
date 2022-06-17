# Map random points on a grid using 2D quadratic interpolation

include("/home/faruk/Git/exploding_brains/wip/core.jl")

import Distributions

OUTDIR = "/home/faruk/Documents/test_julia/test-00"

GRID_SIZE = 20
NR_PARTICLES = 16

# =============================================================================
# Create a directory if it does not exist
if !isdir(OUTDIR)
    mkdir(OUTDIR)
end

# =============================================================================
# Initialize particle positions
p_pos = zeros((NR_PARTICLES, 2))
p_pos[1, :] = [ 2.00, 2.00]
p_pos[2, :] = [ 7.25, 2.00]
p_pos[3, :] = [12.50, 2.00]
p_pos[4, :] = [17.75, 2.00]
p_pos[5, :] = [ 2.00, 7.25]
p_pos[6, :] = [ 7.25, 7.25]
p_pos[7, :] = [12.50, 7.25]
p_pos[8, :] = [17.75, 7.25]
p_pos[9, :] = [ 2.00, 12.50]
p_pos[10, :] = [ 7.25, 12.50]
p_pos[11, :] = [12.50, 12.50]
p_pos[12, :] = [17.75, 12.50]
p_pos[13, :] = [ 2.00, 17.75]
p_pos[14, :] = [ 7.25, 17.75]
p_pos[15, :] = [12.50, 17.75]
p_pos[16, :] = [17.75, 17.75]

# Initialize particle values
p_val = ones(NR_PARTICLES)

# Initialize a zero array
grid = zeros((GRID_SIZE, GRID_SIZE))

# =============================================================================
# Map particles to grid
@time grid = particle_to_grid_interpolate_quadratic(p_pos, grid, p_val)

# Save grid as image
save_png(grid, joinpath(OUTDIR, "interpolation.png"), true)

println("Finished.")
