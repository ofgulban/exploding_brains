# Try Julia for generating brain explosions

import NIfTI
import Images
import LinearAlgebra

INPUT = "/home/faruk/Documents/test_julia/T1w_brain.nii.gz"
OUTDIR = "/home/faruk/Documents/test_julia/frames"

NR_ITER = 10

# =============================================================================
# File input
# =============================================================================
# Create a directory if it does not exist
if !isdir(OUTDIR)
    mkdir(OUTDIR)
end

# Load nifti
nii = NIfTI.niread(INPUT)
NIfTI.niwrite("/home/faruk/Documents/test_julia/T1w_brain_julia.nii.gz", nii)

# Select a slice
data = nii.raw[:, 160, :]
println(string("Slice dims: ", size(data), ", with type: ", typeof(data)))

# Save a PNG image
data = data / maximum(data) * 255  # Normalize to 0-1 range
img = floor.(UInt8, data)
Images.save(joinpath(OUTDIR, "REFERENCE.png"), img)

println("Finished.")
