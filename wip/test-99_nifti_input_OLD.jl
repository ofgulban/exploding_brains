# Try Julia for generating brain explosions

include("/home/faruk/Git/exploding_brains/wip/core.jl")

INPUT = "/home/faruk/Documents/test_julia/T1w_brain.nii.gz"
OUTDIR = "/home/faruk/Documents/test_julia/test-02"

NR_ITER = 24 * 5
DT = 1
GRAVITY = 0.5

# =============================================================================
# File input/output
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
save_png(data, joinpath(OUTDIR, "input.png"))

# =============================================================================
# Prepare particles and velocity field
# =============================================================================
# Initialize particles
p_idx = findall(data .> 0)
p_pos = Array{Float32}(getindex.(p_idx, [1 2])) .- 1

# Record voxel intensity values into particles
p_vals = data[p_idx]

# -----------------------------------------------------------------------------
# Generate a radial velocity field
dims = size(data)
g_velo = zeros(Float32, (dims[1], dims[2], 2))
g_norm = zeros(Float32, dims)
vector = Array{Float32}(undef, 2)
for x = 1:dims[1]
    for y = 1:dims[2]
        # Compute x and y elements, adjust origin in case of even numbers
        if dims[1] % 2 == 0
            vector[1] = x - dims[1]/2 + 0.5
        else
            vector[1] = x - dims[1]/2
        end

        if dims[2] % 2 == 0
            vector[2] = y - dims[2]/2 + 0.5
        else
            vector[1] = x - dims[1]/2
        end

        # Compute L2 norm
        norm = LinearAlgebra.norm(vector)
        # Normalize with norm
        if norm > 0
            vector ./= norm
        end

        # Assign values to image cells
        g_norm[x, y] = norm
        g_velo[x, y, :] = vector
    end
end

global p_velo = g_velo[p_idx, :]

# Save velocity field quality control images
save_png(g_norm, joinpath(OUTDIR, "velocity-norm_initial.png"))
save_png(g_velo[:, :, 1], joinpath(OUTDIR, "velocity-x.png"))
save_png(g_velo[:, :, 2], joinpath(OUTDIR, "velocity-y.png"))

# -----------------------------------------------------------------------------
# Start simulation
# -----------------------------------------------------------------------------
@time for i = 1:NR_ITER
    p_weights = compute_interpolation_weights_2D(p_pos)
    global img, g_mass = particle_to_grid_2D(dims, p_pos, p_vals, p_weights)
    save_png(img, joinpath(OUTDIR, string("values_", lpad(i, 3, "0"), ".png")))

    global p_pos = update_particle_positions_2D(dims, p_pos, p_velo)
    global g_velo = grid_velocity_update_2D(g_velo, g_mass, DT, GRAVITY)
    global p_velo = update_particle_velocities_2D(p_pos, p_velo, g_velo)
end

println("Finished.")
