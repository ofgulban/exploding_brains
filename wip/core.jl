# Core functions

import NIfTI
import Images
import LinearAlgebra

"""
Save a 2D array as PNG image.
"""
function save_png(img_init, path, normalize::Bool)
    img = copy(img_init)

    if normalize
        # Map to 0-255 range (uint8)
        img .-= minimum(img)
        img ./= maximum(img)
        img .*= 255
    end
    img = clamp.(img, 0, 255)
    img = floor.(UInt8, img)

    img = reverse(img, dims=2)
    img = transpose(img)

    Images.save(path, img)
end


"""
Interpolate particles onto a grid.
"""
function particle_to_grid_interpolate_quadratic(p_pos, grid, p_val)
    nr_particles = size(p_pos)[1]

    # Interpolate each particle to 9 grid cells
    Threads.@threads for i = 1:NR_PARTICLES
        # Get a particle's position
        p_x, p_y = p_pos[i, :]

        # Compute which grid cell particle falls within
        cell_idx_x::Int = floor(p_x)
        cell_idx_y::Int = floor(p_y)

        # Convert grid cell index to cell center coordinate
        cell_center_x = cell_idx_x + 0.5
        cell_center_y = cell_idx_y + 0.5

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

        # Get a particle's value
        v = p_val[i]

        # Interpolate particle values onto the grid
        grid[cell_idx_x-1, cell_idx_y-1] += wx_i * wy_i * v
        grid[cell_idx_x  , cell_idx_y-1] += wx_j * wy_i * v
        grid[cell_idx_x+1, cell_idx_y-1] += wx_k * wy_i * v
        grid[cell_idx_x-1, cell_idx_y]   += wx_i * wy_j * v
        grid[cell_idx_x  , cell_idx_y]   += wx_j * wy_j * v
        grid[cell_idx_x+1, cell_idx_y]   += wx_k * wy_j * v
        grid[cell_idx_x-1, cell_idx_y+1] += wx_i * wy_k * v
        grid[cell_idx_x  , cell_idx_y+1] += wx_j * wy_k * v
        grid[cell_idx_x+1, cell_idx_y+1] += wx_k * wy_k * v
    end
    return grid
end


"""
Clamp particles within bounds
"""
function clamp_and_bounce(p_pos, p_vel, grid_dim_x, grid_dim_y)
    nr_particles = size(p_pos)[1]

    Threads.@threads for i = 1:nr_particles
        if p_pos[i, 1] < 3-0.5
            p_pos[i, 1] = 3-0.5
            p_vel[i, 1] *= -1
        elseif p_pos[i, 1] > grid_dim_x-0.5
            p_pos[i, 1] = grid_dim_x-0.5
            p_vel[i, 1] *= -1
        end

        if p_pos[i, 2] < 3-0.5
            p_pos[i, 2] = 3-0.5
            p_vel[i, 2] *= -1
        elseif p_pos[i, 2] > grid_dim_y-0.5
            p_pos[i, 2] = grid_dim_y-0.5
            p_vel[i, 2] *= -1
        end
    end
    return p_pos, p_vel
end


"""
Interpolate particles onto a grid with more complicated model.
"""
function particle_to_grid_velocity(p_pos, grid, p_vel, p_mass, p_C, p_Fs, p_vol)
    local nr_particles = size(p_pos)[1]
    local g_vel = zeros((size(grid)[1], size(grid)[2], 2))
    local g_mass = zeros((size(grid)[1], size(grid)[2]))

    # Lamé parameters for stress-strain relationship
    ELASTIC_LAMBDA = 10.0
    ELASTIC_MU = 20.0

    # Interpolate each particle to 9 grid cells
    Threads.@threads for i = 1:nr_particles
        # ---------------------------------------------------------------------
        # NOTE[Faruk]: This section is a direct translation from:
        # https://github.com/nialltl/incremental_mpm/blob/2c4659230ec1added189913da11ed67f182f5b4d/Assets/2.%20MLS_MPM_NeoHookean_Multithreaded/MLS_MPM_NeoHookean_Multithreaded.cs#L236-L309
        # I need to digest the maths a bit to re-think the implementation.
        F = p_Fs[i]
        J = LinearAlgebra.det(F)
        # MPM course, page 46
        volume = p_vol[i] * J

        # Useful matrices for Neo-Hookean model
        F_T = LinearAlgebra.transpose(F)
        F_inv_T = LinearAlgebra.inv(F_T)
        F_minus_F_inv_T = F - F_inv_T

        # MPM course equation 48
        P_term_0 = ELASTIC_MU * F_minus_F_inv_T
        P_term_1 = ELASTIC_LAMBDA * log(J) * F_inv_T
        P = P_term_0 + P_term_1

        # Cauchy_stress = (1 / det(F)) * P * F_T (equation 38, MPM course)
        stress = (1.0 / J) * P * F_T

        # (M_p)^-1 = 4, see APIC paper and MPM course page 42
        # this term is used in MLS-MPM paper eq. 16. with quadratic weights,
        # Mp = (1/4) * (delta_x)^2. in this simulation, delta_x = 1, because
        # i scale the rendering of the domain rather than the domain itself.
        # we multiply by dt as part of the process of fusing the momentum and
        # force update for MLS-MPM
        eq_16_term_0 = -volume * 4 * stress * TIME_STEP

        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # NOTE[Faruk]: I should double check this section.
        # Compute cell distances to particle
        cell_dist_x_i = p_x - (cell_center_x - 1)
        cell_dist_x_j = p_x - cell_center_x
        cell_dist_x_k = p_x - (cell_center_x + 1)

        cell_dist_y_i = p_y - (cell_center_y - 1)
        cell_dist_y_j = p_y - cell_center_y
        cell_dist_y_k = p_y - (cell_center_y + 1)

        # Compute additional grid data
        Q_x_i = p_C[i] * cell_dist_x_i
        Q_x_j = p_C[i] * cell_dist_x_j
        Q_x_k = p_C[i] * cell_dist_x_k

        Q_y_i = p_C[i] * cell_dist_y_i
        Q_y_j = p_C[i] * cell_dist_y_j
        Q_y_k = p_C[i] * cell_dist_y_k
        # ---------------------------------------------------------------------
        # Fused force/momentum update from MLS-MPM
        # see MLS-MPM paper, equation listed after eqn. 28
        g_vel[g_x-1, g_y-1, 1] += wx_i * wy_i * p_mass[i] * (p_vel[i, 1] + Q_x_i) + wx_i * wy_i * eq_16_term_0 * cell_dist_x_i
        g_vel[g_x  , g_y-1, 1] += wx_j * wy_i * p_mass[i] * (p_vel[i, 1] + Q_x_j) + wx_j * wy_i * eq_16_term_0 * cell_dist_x_j
        g_vel[g_x+1, g_y-1, 1] += wx_k * wy_i * p_mass[i] * (p_vel[i, 1] + Q_x_k) + wx_k * wy_i * eq_16_term_0 * cell_dist_x_k
        g_vel[g_x-1, g_y, 1]   += wx_i * wy_j * p_mass[i] * (p_vel[i, 1] + Q_x_i) + wx_i * wy_j * eq_16_term_0 * cell_dist_x_i
        g_vel[g_x  , g_y, 1]   += wx_j * wy_j * p_mass[i] * (p_vel[i, 1] + Q_x_j) + wx_j * wy_j * eq_16_term_0 * cell_dist_x_j
        g_vel[g_x+1, g_y, 1]   += wx_k * wy_j * p_mass[i] * (p_vel[i, 1] + Q_x_k) + wx_k * wy_j * eq_16_term_0 * cell_dist_x_k
        g_vel[g_x-1, g_y+1, 1] += wx_i * wy_k * p_mass[i] * (p_vel[i, 1] + Q_x_i) + wx_i * wy_k * eq_16_term_0 * cell_dist_x_i
        g_vel[g_x  , g_y+1, 1] += wx_j * wy_k * p_mass[i] * (p_vel[i, 1] + Q_x_j) + wx_j * wy_k * eq_16_term_0 * cell_dist_x_j
        g_vel[g_x+1, g_y+1, 1] += wx_k * wy_k * p_mass[i] * (p_vel[i, 1] + Q_x_k) + wx_k * wy_k * eq_16_term_0 * cell_dist_x_k

        g_vel[g_x-1, g_y-1, 2] += wx_i * wy_i * p_mass[i] * (p_vel[i, 2] + Q_y_i) + wx_i * wy_i * eq_16_term_0 * cell_dist_y_i
        g_vel[g_x  , g_y-1, 2] += wx_j * wy_i * p_mass[i] * (p_vel[i, 2] + Q_y_i) + wx_j * wy_i * eq_16_term_0 * cell_dist_y_i
        g_vel[g_x+1, g_y-1, 2] += wx_k * wy_i * p_mass[i] * (p_vel[i, 2] + Q_y_i) + wx_k * wy_i * eq_16_term_0 * cell_dist_y_i
        g_vel[g_x-1, g_y, 2]   += wx_i * wy_j * p_mass[i] * (p_vel[i, 2] + Q_y_j) + wx_i * wy_j * eq_16_term_0 * cell_dist_y_j
        g_vel[g_x  , g_y, 2]   += wx_j * wy_j * p_mass[i] * (p_vel[i, 2] + Q_y_j) + wx_j * wy_j * eq_16_term_0 * cell_dist_y_j
        g_vel[g_x+1, g_y, 2]   += wx_k * wy_j * p_mass[i] * (p_vel[i, 2] + Q_y_j) + wx_k * wy_j * eq_16_term_0 * cell_dist_y_j
        g_vel[g_x-1, g_y+1, 2] += wx_i * wy_k * p_mass[i] * (p_vel[i, 2] + Q_y_k) + wx_i * wy_k * eq_16_term_0 * cell_dist_y_k
        g_vel[g_x  , g_y+1, 2] += wx_j * wy_k * p_mass[i] * (p_vel[i, 2] + Q_y_k) + wx_j * wy_k * eq_16_term_0 * cell_dist_y_k
        g_vel[g_x+1, g_y+1, 2] += wx_k * wy_k * p_mass[i] * (p_vel[i, 2] + Q_y_k) + wx_k * wy_k * eq_16_term_0 * cell_dist_y_k

        # Compute mass grid
        g_mass[g_x-1, g_y-1] += wx_i * wy_i * p_mass[i]
        g_mass[g_x  , g_y-1] += wx_j * wy_i * p_mass[i]
        g_mass[g_x+1, g_y-1] += wx_k * wy_i * p_mass[i]
        g_mass[g_x-1, g_y]   += wx_i * wy_j * p_mass[i]
        g_mass[g_x  , g_y]   += wx_j * wy_j * p_mass[i]
        g_mass[g_x+1, g_y]   += wx_k * wy_j * p_mass[i]
        g_mass[g_x-1, g_y+1] += wx_i * wy_k * p_mass[i]
        g_mass[g_x  , g_y+1] += wx_j * wy_k * p_mass[i]
        g_mass[g_x+1, g_y+1] += wx_k * wy_k * p_mass[i]
    end

    # Normalize momentum grid with mass grid to attain velocity grid
    idx = g_mass .> 0
    g_vel[idx, 1] ./= g_mass[idx]
    g_vel[idx, 2] ./= g_mass[idx]

    return g_vel
end

"""
Grid to particle mapping
"""
function grid_to_particle_estimate_volume(p_pos, g_mass, p_mass)
    local nr_particles = size(p_pos)[1]
    local p_density = zeros(size(p_mass))

    # Grid to particle mapping
    for i = 1:nr_particles
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

        # Map from grid to particle
        p_density[i, 1] = 0
        p_density[i, 1] += wx_i * wy_i * g_mass[g_x-1, g_y-1]
        p_density[i, 1] += wx_j * wy_i * g_mass[g_x  , g_y-1]
        p_density[i, 1] += wx_k * wy_i * g_mass[g_x+1, g_y-1]
        p_density[i, 1] += wx_i * wy_j * g_mass[g_x-1, g_y  ]
        p_density[i, 1] += wx_j * wy_j * g_mass[g_x  , g_y  ]
        p_density[i, 1] += wx_k * wy_j * g_mass[g_x+1, g_y  ]
        p_density[i, 1] += wx_i * wy_k * g_mass[g_x-1, g_y+1]
        p_density[i, 1] += wx_j * wy_k * g_mass[g_x  , g_y+1]
        p_density[i, 1] += wx_k * wy_k * g_mass[g_x+1, g_y+1]
    end
    return p_mass ./ p_density
end

# =============================================================================
# NOTE: Below functions either needs rework or deprecation
# =============================================================================
"""
Compute interpolation weights using particles on a grid.
NOTE: Marked for deprecation
"""
function compute_interpolation_weights_2D(p_pos)
    nr_particles = size(p_pos)[1]
    p_weights = zeros(Float32, nr_particles, 2, 2)  # n × 2 × 2

    Threads.@threads for n = 1:nr_particles
        # Particle coordinates
        x = p_pos[n, 1]
        y = p_pos[n, 2]

        # Binned particle coordinates
        i = floor(Int32, x)
        j = floor(Int32, y)

        # 4 nearest neighbor weights (bilinear)
        dx = abs(x - i)
        dy = abs(y - j)
        p_weights[n, 1, 1] = dx*dy

        dx = abs(x - i + 1)
        dy = abs(y - j)
        p_weights[n, 1, 2] = dx*dy

        dx = abs(x - i)
        dy = abs(y - j + 1)
        p_weights[n, 2, 1] = dx*dy

        dx = abs(x - i + 1)
        dy = abs(y - j + 1)
        p_weights[n, 2, 2] = dx*dy
    end
    return p_weights
end


"""
Fill in a 2D image using particle positions and values.
"""
function particle_to_grid_2D(dims, p_pos, p_vals, p_weights)
    nr_particles = size(p_pos)[1]
    local img_vals = zeros(Float32, dims)
    local img_mass = zeros(Float32, dims)

    for n = 1:nr_particles
        # Ignore out of bounds particles
        if p_pos[n, 1] < 1
        elseif p_pos[n, 1] >= dims[1]
        elseif p_pos[n, 2] < 1
        elseif p_pos[n, 2] >= dims[2]
        else
            # Bin particle positions to pixel indices
            i = floor(Int32, p_pos[n, 1])
            j = floor(Int32, p_pos[n, 2])

            # Fill-in image with weighted particle values (precomputed 2x2)
            img_vals[i, j] += p_weights[n, 1, 1] * p_vals[n]
            img_vals[i, j+1] += p_weights[n, 1, 2] * p_vals[n]
            img_vals[i+1, j] += p_weights[n, 2, 1] * p_vals[n]
            img_vals[i+1, j+1] += p_weights[n, 2, 2] * p_vals[n]

            # Records weights as a mass field
            img_mass[i, j] += p_weights[n, 1, 1]
            img_mass[i, j+1] += p_weights[n, 1, 2]
            img_mass[i+1, j] += p_weights[n, 2, 1]
            img_mass[i+1, j+1] += p_weights[n, 2, 2]
        end
    end

    # Normalize with mass for voxels that has multiple particles
    idx = findall(img_mass .> 0)
    img_vals[idx] ./= img_mass[idx]

    return img_vals, img_mass
end


"""
Update particle position using particle velocities
"""
function update_particle_positions_2D(dims, p_pos, p_velo)
    nr_particles = size(p_pos)[1]
    for n = 1:nr_particles
        # Update particle positions
        p_pos[n, 1] += p_velo[n, 1]
        p_pos[n, 2] += p_velo[n, 2]

        # ---------------------------------------------------------------------
        # NOTE: Not sure how useful it is to check this here
        # ---------------------------------------------------------------------
        # Check for the boundary conditionss
        if p_pos[n, 1] < 1
            p_pos[n, 1] = 1
        elseif p_pos[n, 1] >= dims[1]-1
            p_pos[n, 1] = dims[1]-1
        end

        if p_pos[n, 2] < 1
            p_pos[n, 2] = 1
        elseif p_pos[n, 2] >= dims[2]-1
            p_pos[n, 2] = dims[2]-1
        end
    end
    return p_pos
end


"""
Update particle velocities based on velocity grid.
"""
function update_particle_velocities_2D(p_pos, p_velo, g_velo)
    nr_particles = size(p_pos)[1]

    for n = 1:nr_particles
        # Bin particle positions to 2D grid indices
        i = floor(Int32, p_pos[n, 1])
        j = floor(Int32, p_pos[n, 2])

        # Update particle velocities based on the velocity 2D grid
        p_velo[n, 1] = g_velo[i, j, 1]
        p_velo[n, 2] = g_velo[i, j, 2]
    end
    return p_velo
end


"""
Update grid velocities
"""
function grid_velocity_update_2D(g_velo, g_mass, dt, gravity)
    g_idx = findall(g_mass .> 0)

    # Dampen velocities by mass
    g_velo[g_idx, :] ./= g_mass[g_idx]

    # Adjust for gravity and time step
    g_velo[g_idx, :] .+= dt * gravity

    return g_velo
end

# """
# Compute a scalar field using particles.
# """
# function particle_to_grid(p_pos, p_C, p_mass, p_velo, cells, p_weights, p_vals)
#     dims = size(cells)
#     nr_particles = size(p_velo)[1]
#
#     c_mass = zeros(Float32, dims)  # scalar field
#     c_velo = zeros(Float32, dims[1], dims[2], 2)  # vector field
#     c_values = zeros(Float32, dims)  # scalar field
#
#     # for i = 1:nr_particles
#     Threads.@threads for i = 1:nr_particles
#         p = p_pos[i, :]  # particle coordinates
#         C = p_C[i, :, :]  # TODO: What is this variable?
#         m = p_mass[i]  # particle masses
#         v = p_velo[i, :]  # particle velocities
#         w = p_weights[i, :, :]  # particle neighbour interpolation weights
#         value = p_vals[i]  # particle values
#
#         # 9 cell neighbourhood of the particle
#         p_idx = floor.(Int32, p)
#         for gx = Int32(1):Int32(3)
#             for gy = Int32(1):Int32(3)
#                 weight = w[gx, 1] * w[gy, 2]
#
#                 idx = p_idx[1] + gx - Int32(1), p_idx[2] + gy - Int32(1)
#                 cell_dist = (idx .- p) .+ Float32(0.5)
#                 Q1 = LinearAlgebra.dot(C[1, :], cell_dist)
#                 Q2 = LinearAlgebra.dot(C[2, :], cell_dist)
#                 Q = Q1 + Q2
#
#                 # MPM course equation 172
#                 mass_contrib = weight * m
#
#                 # Insert into grid
#                 c_mass[idx[1], idx[2]] += mass_contrib
#                 c_velo[idx[1], idx[2], :] += mass_contrib .* (v .+ Q)
#
#                 # For carrying voxel values (grayscale image)
#                 value_contrib = weight * value
#                 c_values[idx[1], idx[2]] += value_contrib
#
#                 # NOTE: Cell velocity is actually momentum here. It will be
#                 # updated later.
#             end
#         end
#     end
#     return c_mass, c_velo, c_values
# end
#
#
# """
# Operate on velocity grid.
# """
# function grid_velocity_update(c_velo, c_mass, dt, gravity)
#     idx = findall(c_mass .> 0)
#     c_velo[idx, 1] ./= c_mass[idx]
#     c_velo[idx, 2] ./= c_mass[idx]
#     c_velo[idx, 1] .+= dt * gravity  # Gravity only effects y
#     return c_velo
# end
#
#
# """
# Update particles based on velocities on the grid.
# """
# function grid_to_particle_velocity(p_pos, p_velo, p_weights, c_velo, dt,
#                                    rule, bounce_factor)
#     dims = size(c_velo)[1]
#     nr_part = size(p_pos)[1]
#     # Reset particle velocity
#     p_velo .*= 0
#
#     for i =1:nr_part
#         p = p_pos[i, :]
#         v = p_velo[i, :]
#         w = p_weights[i, :, :]
#
#         # Construct affine per-particle momentum matrix from (APIC)/MLS-MPM.
#         B = zeros(Float32, 2, 2)
#
#         # 9 cell neighbourhood of the particle
#         p_idx = floor.(Int32, p)
#         for gx = Int32(1):Int32(3)
#             for gy = Int32(1):Int32(3)
#                 weight = w[gx, 1] * w[gy, 2]
#
#                 idx = p_idx[1] + gx - Int32(1), p_idx[2] + gy - Int32(1)
#                 cell_dist = (idx .- p) .+ Float32(0.5)
#                 weighted_velocity = c_velo[idx[1], idx[2], :] .* weight
#
#                 # APIC paper equation 10, constructing inner term for B
#                 term = weighted_velocity .* cell_dist
#
#                 B .+= term
#                 v .+= weighted_velocity
#             end
#         end
#
#         # p_C[i] = B * 4  # unused for now
#
#         # Advect particles
#         p .+= v .* dt
#
#         # Act on escaped particles
#         # p, v = clamp(p, v, d_min=0, d_max=dims[1], rule="bounce",
#         #              bounce_factor=bounce_factor)
#
#         # Update particles
#         p_pos[i, :] = p[:]
#         p_velo[i, :] = v[:]
#     end
#     return p_pos, p_velo
# end
