import numpy as np
from itertools import product
import pygame
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt

mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)

rng = np.random.default_rng()
im = complex(0, 1)


def main():
    real_field_simulation()
    plot_action()
    Phi = np.load('real_minus_m_1p3_lambda_100p0_40b15b15b15.npy')
    visualize_2d_slice(Phi, delay=50, scale_factor=20)
    plot_correlations(Phi, 15/15, 5/40)

    avg, std = real_monte_carlo(Phi, 15/15, 5/40, real_potential, 1.3, 1.8, 10, 5, time_correlation)
    np.save('real_minus_m_1p3_lambda_100p0_40b15b15b15_TimeCorrelation', avg)
    np.save('real_minus_m_1p3_lambda_100p0_40b15b15b15_TimeCorrelation_std', std)
    perform_fit()


def real_field_simulation():
    size = 15
    time_steps = 40
    spacing = 15 / size  # So each side is of 15 units total length every time
    time_spacing = 5 / time_steps  # So the simulation runs for 5 time units every time

    Phi = initialize_real_lattice((size, size, size), time_steps)
    time_steps, x_size, y_size, z_size = Phi.shape
    print("\nShape of Phi:", Phi.shape)
    print(x_size * y_size * z_size * time_steps, "lattice points")

    num_sweeps = 80  # Average number of times each lattice point will be updated
    m = 1.3
    lam = 100
    num_updates, Action = real_metropolis(Phi, spacing, time_spacing, real_potential_minus_m, m, lam, num_sweeps)
    np.save('real_minus_m_1p3_lambda_100p0_40b15b15b15', Phi)
    np.save('real_minus_m_1p3_lambda_100p0_40b15b15b15_Action', Action)


def perform_fit():
    corr = np.load('real_minus_m_1p3_lambda_100p0_40b15b15b15_TimeCorrelation.npy')
    std = np.load('real_minus_m_1p3_lambda_100p0_40b15b15b15_TimeCorrelation_std.npy')
    time_steps = len(corr)
    plt.errorbar(np.arange(time_steps) * 5 / 40, corr, yerr=std, label="Averaged Correlation", ls=" ")
    plt.xlim(0, 5)
    (p, C, info1, info2, info3) = opt.curve_fit(exponential_fit, np.arange(0, time_steps)[2:] * 5 / 40, corr[2:],
                                                sigma=std[2:], p0=1.0, absolute_sigma=True, full_output=True)
    err = np.sqrt(np.diag(C))
    print(p, err)
    nums = np.linspace(0, time_steps * 5 / 40, 1000)
    plt.plot(nums, exponential_fit(nums, p), label="Best Fit")
    plt.title("Averaged Correlation")
    plt.xlabel("Distance")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()


def exponential_fit(x, m):
    return np.exp(-m*x)


def real_monte_carlo(Phi, spacing, time_spacing, potential, m, lam, measurements, measurement_spacing, func):
    """
    Given a lattice already in equilibrium/low action.
    measurement_spacing: Number of Monte Carlo sweeps between each measurement (one sweep = as many iterations as
    there are lattice points). This should be high enough that the lattice looks different each time.
    measurements: Total measurements of the quantity to take.

    Returns: The average values and standard deviations
    """
    time_steps, x_size, y_size, z_size = Phi.shape
    vals = np.zeros((measurements, time_steps))  # Currently tracking time correlation function
    for i in range(0, measurements):
        for j in range(0, time_steps):
            vals[i, j] = func(Phi, j)
        vals[i, :] /= vals[i, 0]  # This line is specifically to normalize the correlation function, remove it for other functions
        print("Monte Carlo sweep", i, "done")
        real_metropolis(Phi, spacing, time_spacing, potential, m, lam, measurement_spacing)
    averaged_vals = np.zeros(time_steps)
    std = np.zeros(time_steps)
    for j in range(0, time_steps):
        averaged_vals[j] = np.mean(vals[:, j])
        std[j] = np.std(vals[:, j])
    return averaged_vals, std


def plot_correlations(Phi, spacing, time_spacing):
    time_steps, x_size, y_size, z_size = Phi.shape
    corr = np.zeros(time_steps)
    for i in range(0, time_steps):
        corr[i] = time_correlation(Phi, i)
    corr /= corr[0]
    plt.plot(np.arange(time_steps) * time_spacing, corr)
    plt.xlim(0, 5)
    plt.title("Correlation Function for One Configuration")
    plt.xlabel("Separation")
    plt.ylabel("Correlation")
    plt.show()


def plot_action():
    Actions = np.load('real_minus_m_1p3_lambda_100p0_40b15b15b15_Action.npy')
    Sweeps = np.arange(len(Actions))
    plt.plot(Sweeps, np.log10(Actions))
    plt.xlim(0, len(Sweeps) - 1)
    # plt.ylim(0, Actions[0])
    plt.title("Action Throughout Metropolis Algorithm")
    plt.xlabel("Sweep Number")
    plt.ylabel("log(Action)")
    plt.show()


def initialize_real_lattice(sizes, time_steps):
    """
    sizes: tuple of spatial dimensions
    time_steps: number of spatial dimensions
    """
    x_size, y_size, z_size = sizes
    Phi = (2 * np.random.rand(time_steps, x_size, y_size, z_size) - 1.)
    return Phi


def time_correlation(Phi, distance):
    """
       Two point time correlation function for a certain distance between the two points
    """
    size_t, size_x, size_y, size_z = Phi.shape
    correlations = []

    # Iterate over all points in the lattice at t = 0
    t = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                t_shifted = (t + distance)
                value = Phi[t, x, y, z] * Phi[t_shifted, x, y, z]
                correlations.append(value)
    return np.mean(correlations)


def space_correlation(Phi, distance):
    """
    Two point spatial correlation function for a certain distance between the two points
    """
    size_t, size_x, size_y, size_z = Phi.shape
    correlations = []

    # Iterate over all points in the lattice
    for t in range(size_t):
        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    for mu in range(1, 4):
                        value = 0
                        value_minus = 0
                        if mu == 1:  # x direction
                            x_shifted = (x + distance) % size_x
                            x_shifted_minus = (x - distance) % size_x
                            value = Phi[t, x, y, z] * Phi[t, x_shifted, y, z]
                            value_minus = Phi[t, x, y, z] * Phi[t, x_shifted_minus, y, z]
                        elif mu == 2:  # y direction
                            y_shifted = (y + distance) % size_y
                            y_shifted_minus = (y - distance) % size_y
                            value = Phi[t, x, y, z] * Phi[t, x, y_shifted, z]
                            value_minus = Phi[t, x, y, z] * Phi[t, x, y_shifted_minus, z]
                        elif mu == 3:  # z direction
                            z_shifted = (z + distance) % size_z
                            z_shifted_minus = (z - distance) % size_z
                            value = Phi[t, x, y, z] * Phi[t, x, y, z_shifted]
                            value_minus = Phi[t, x, y, z] * Phi[t, x, y, z_shifted_minus]
                        correlations.append(value)
                        correlations.append(value_minus)
    return np.mean(correlations)


def real_metropolis(Phi, spacing, time_spacing, potential_func, m, lam, num_sweeps, delta_std=0.1):
    """
    Perform a Monte Carlo sweep by repeatedly proposing random updates to the field.
    Returns the number of accepted updates, and an array of the action after each full sweep
    """
    time_steps, x_size, y_size, z_size = Phi.shape
    num_attempts = num_sweeps * time_steps * x_size * y_size * z_size
    accepted_updates = 0
    steps = 0
    print(" ")
    Action = np.zeros(num_sweeps + 1)
    Action[0] = calculate_real_action(Phi, spacing, time_spacing, potential_func, m, lam)

    for _ in range(num_attempts):
        random_index = tuple(np.random.randint(dim) for dim in Phi.shape)
        delta_real = np.random.normal(0, delta_std)  # Normally distributed random change

        if update_real_field(Phi, random_index, delta_real, spacing, time_spacing, potential_func, m, lam):
            accepted_updates += 1
        steps += 1
        if steps % 1000 == 0:
            progress = steps / num_attempts
            print("\r", end="")
            print(f"Simulating... {np.round(100 * progress, 2)}%", end="")
        if steps % (time_steps * x_size * y_size * z_size) == 0:
            Action[int(np.round(steps / (time_steps * x_size * y_size * z_size)))] = (
                calculate_real_action(Phi, spacing, time_spacing, potential_func, m, lam))
    print("\n")
    return accepted_updates, Action


def update_real_field(Phi, index, delta, spacing, time_spacing, potential_func, m, lam):
    action_change = delta_S(Phi, index, delta, spacing, time_spacing, potential_func, m, lam)
    if action_change < 0:
        Phi[index] += delta
        return True
    else:
        if rng.random() < np.exp(-action_change):
            Phi[index] += delta
            return True
        return False


def delta_S(Phi, index, delta, spacing, time_spacing, potential_func, m, lam):
    t, x, y, z = index
    time_steps, x_size, y_size, z_size = Phi.shape

    prev_x = (x - 1) % x_size
    prev_y = (y - 1) % y_size
    prev_z = (z - 1) % z_size

    initial_value = Phi[index]

    current_action = 0
    derivative_sum = 0.0
    if t != time_steps - 1:
        t_der = discrete_derivative(Phi, index, 0, spacing, time_spacing)
        derivative_sum += t_der**2
    for mu in range(1, 4):
        derivative = discrete_derivative(Phi, index, mu, spacing, time_spacing)
        derivative_sum += derivative ** 2
    current_action += 0.5 * derivative_sum

    derivative_sum = 0  # Reset it for this part
    if t != 0:
        t_der = discrete_derivative(Phi, (t-1, x, y, z), 0, spacing, time_spacing)
        derivative_sum += t_der**2
    x_der = discrete_derivative(Phi, (t, prev_x, y, z), 1, spacing, time_spacing)
    derivative_sum += x_der**2
    y_der = discrete_derivative(Phi, (t, x, prev_y, z), 2, spacing, time_spacing)
    derivative_sum += y_der**2
    z_der = discrete_derivative(Phi, (t, x, y, prev_z), 3, spacing, time_spacing)
    derivative_sum += z_der**2
    current_action += 0.5 * derivative_sum

    current_action += potential_func(Phi[index], m, lam)

    Phi[index] += delta

    proposed_action = 0

    derivative_sum = 0.0
    if t != time_steps - 1:
        t_der = discrete_derivative(Phi, index, 0, spacing, time_spacing)
        derivative_sum += t_der ** 2
    for mu in range(1, 4):
        derivative = discrete_derivative(Phi, index, mu, spacing, time_spacing)
        derivative_sum += derivative ** 2
    proposed_action += 0.5 * derivative_sum

    derivative_sum = 0  # Reset it for this part
    if t != 0:
        t_der = discrete_derivative(Phi, (t - 1, x, y, z), 0, spacing, time_spacing)
        derivative_sum += t_der ** 2
    x_der = discrete_derivative(Phi, (t, prev_x, y, z), 1, spacing, time_spacing)
    derivative_sum += x_der ** 2
    y_der = discrete_derivative(Phi, (t, x, prev_y, z), 2, spacing, time_spacing)
    derivative_sum += y_der ** 2
    z_der = discrete_derivative(Phi, (t, x, y, prev_z), 3, spacing, time_spacing)
    derivative_sum += z_der ** 2
    proposed_action += 0.5 * derivative_sum

    proposed_action += potential_func(Phi[index], m, lam)

    Phi[index] = initial_value

    return proposed_action - current_action


def calculate_real_action(Phi, spacing, time_spacing, potential_func, m, lam):
    time_steps, x_size, y_size, z_size = Phi.shape
    action = 0.0

    # Iterate over each lattice point
    for t in range(0, time_steps - 1):
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    index = (t, x, y, z)
                    derivative_sum = 0.0
                    for mu in range(4):  # Directions of derivative: mu = 0 (time), = 1 (x), = 2 (y), = 3 (z)
                        derivative = discrete_derivative(Phi, index, mu, spacing, time_spacing)
                        derivative_sum += derivative**2
                    action += 0.5 * derivative_sum + potential_func(Phi[index], m, lam)

    action *= spacing ** 3 * time_spacing
    return action


def discrete_derivative(Phi, index, direction, spacing, time_spacing):
    """
    Computes the discrete derivative with periodic boundary conditions in the spatial dimensions.
    direction (int): The axis along which to compute the derivative (0 for time, 1-3 for space).
    """
    t, x, y, z = index
    shape = Phi.shape

    if direction == 0:  # Time derivative (no looping around)
        if t + 1 < shape[0]:
            next_index = (t + 1, x, y, z)
        else:
            raise IndexError("Index out of bounds for time derivative")
    elif direction == 1:
        next_index = (t, (x + 1) % shape[1], y, z)
    elif direction == 2:
        next_index = (t, x, (y + 1) % shape[2], z)
    elif direction == 3:
        next_index = (t, x, y, (z + 1) % shape[3])
    else:
        raise ValueError("Invalid direction. Must be 0 (time) or 1, 2, 3 (space).")

    if direction == 0:
        derivative = (Phi[next_index] - Phi[index]) / time_spacing
    else:
        derivative = (Phi[next_index] - Phi[index]) / spacing

    return derivative


def brute_force_partition_function(Phi, spacing, time_spacing, potential_func, m, lam, phi_max, num_samples):
    """
    Calculate the partition function Z using brute force integration over field values from -phi_max to phi_max,
    split into num_samples parts.
    This very quickly becomes infeasible to calculate directly.
    """
    time_steps, x_size, y_size, z_size = Phi.shape
    total_sites = time_steps * x_size * y_size * z_size

    phi_values = np.linspace(-phi_max, phi_max, num_samples)
    delta_phi = phi_values[1] - phi_values[0]  # Spacing between samples

    Z = 0.0

    # Iterating over all possible field configurations
    for field_configuration in product(phi_values, repeat=total_sites):
        field_array = np.array(field_configuration).reshape(Phi.shape)
        action = calculate_real_action(field_array, spacing, time_spacing, potential_func, m, lam)
        Z += np.exp(-action)

    Z *= delta_phi ** total_sites

    return Z


def real_potential(field_value, m, lam):
    """
    Self-interacting potential for a real field, positive m^2
    Set lam = 0 to get a free scalar field, which should obey the Klein-Gordon equation
    """
    return (1/2.) * m**2 * field_value**2 + (1. / (4*3*2*1)) * lam * field_value**4


def real_potential_minus_m(field_value, m, lam):
    """
    Self-interacting potential for a real field, negative m^2
    Set lam = 0 to get a free scalar field, which should obey the Klein-Gordon equation
    """
    return -(1/2.) * m**2 * field_value**2 + (1. / (4*3*2*1)) * lam * field_value**4


def visualize_2d_slice(Phi, slice_z=0, scale_factor=10, delay=100):
    """
    slice_z (int): The z-index for the 2D slice to visualize
    scale_factor (int): Size of each lattice point on the screen
    delay: length of each frame in milliseconds
    """
    time_steps, x_size, y_size, z_size = Phi.shape

    pygame.init()
    screen_width, screen_height = x_size * scale_factor, y_size * scale_factor
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("2D Slice of Scalar Field")

    # Main loop over time steps
    running = True
    for t in range(time_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        slice_2d = Phi[t, :, :, slice_z]

        squared_magnitude = np.abs(slice_2d)**2
        max_val = np.max(squared_magnitude)  # Max value, for normalization

        for x in range(x_size):
            for y in range(y_size):
                value = squared_magnitude[x, y]
                gray_value = 255 - int((value / max_val) * 255) if max_val > 0 else 255
                color = (gray_value, gray_value, gray_value)

                pygame.draw.rect(
                    screen,
                    color,
                    (x * scale_factor, y * scale_factor, scale_factor, scale_factor)
                )

        pygame.display.flip()
        pygame.time.delay(delay)

    pygame.quit()


if __name__ == '__main__':
    main()
