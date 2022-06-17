# Try Julia for generating brain explosions

include("/home/faruk/Git/exploding_brains/wip/core.jl")

INPUT = "/home/faruk/Documents/test_julia/T1w_brain.nii.gz"
OUTDIR = "/home/faruk/Documents/test_julia/frames"

NR_ITER = 10
DT = 1  # Time step (smaller = more accurate simulation)
GRAVITY = 0.05

# =============================================================================
# File input
# =============================================================================
# Create a directory if it does not exist
if !isdir(OUTDIR)
    mkdir(OUTDIR)
end

# Load nifti
nii = NIfTI.niread(INPUT)

# Select a slice
data = nii.raw[:, 160, :]
println(string("Slice dims: ", size(data), ", with type: ", typeof(data)))

# Save a PNG image
save_png(data, joinpath(OUTDIR, "REFERENCE.png"))

# =============================================================================
# Particle simulation here
# =============================================================================
# Initialize particles
p_idx = findall(data .> 0)
p_pos = getindex.(p_idx, [1 2]) .- Float32(0.5)

# Record voxel intensity values into particles
p_vals = data[p_idx]

# Initialize velocities
NR_PART = size(p_idx)
p_velo = zeros(Float32, NR_PART[1], 2)
p_velo[:, 1] = rand(Float32, NR_PART) .+ Float32(0.5)
p_velo[:, 2] = rand(Float32, NR_PART) .* Float32(4)

# Initialize masses
p_mass = ones(Float32, NR_PART)

# Initialize C (NOTE: I forgot what this term is. Need to check.)
p_C = zeros(Float32, NR_PART[1], 2, 2)

# Initialize cells
cells = zeros(Float32, size(data))

# -----------------------------------------------------------------------------
# Start simulation
# -----------------------------------------------------------------------------
@time for i = 1:NR_ITER
    p_weights = compute_interpolation_weights(p_pos)
    c_mass, c_velo, c_values = particle_to_grid(
        p_pos, p_C, p_mass, p_velo, cells, p_weights, p_vals)
    c_velo = grid_velocity_update(c_velo, c_mass, DT, GRAVITY)

    global p_pos, p_velo = grid_to_particle_velocity(
        p_pos, p_velo, p_weights, c_velo, DT,
        "bounce", -0.9)

    save_png(c_mass, joinpath(OUTDIR, string("mass_", i, ".png")))
    save_png(c_velo, joinpath(OUTDIR, string("velo_", i, ".png")))
    save_png(c_values, joinpath(OUTDIR, string("values_", i, ".png")))
end

println("Finished.")
