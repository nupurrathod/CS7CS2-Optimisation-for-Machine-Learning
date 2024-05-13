from random import uniform
import matplotlib.pyplot as plt
import numpy as np

def cost_function_one(x, y):
    return 8 * (x - 3) ** 4 + 4 * (y - 1) ** 2

def cost_function_two(x, y):
    return np.maximum(x - 3, 0) + 4 * np.abs(y - 1)

def gradient_one(x, y):
    return np.array([32 * (x - 3)**3, 8 * (y - 1)])

def gradient_two(x, y):
    grad_x = np.where(x > 3, 1, 0)
    grad_y = 4 * np.sign(y - 1)
    return np.array([grad_x, grad_y])
# Random Search
def gs_random(evaluate_cost, dimensions, limits, num_samples):
    minimum_cost = float('inf')
    history_cost = []
    history_minimum_cost = []
    sequence_history = []
    optimal_sequence_history = []

    for _ in range(num_samples):
        trial_sequence = [uniform(limits[i][0], limits[i][1]) for i in range(dimensions)]
        evaluation = evaluate_cost(*trial_sequence)
        sequence_history.append(trial_sequence)
        history_cost.append(evaluation)
        if evaluation < minimum_cost:
            minimum_cost = evaluation
            optimal_sequence_history.append(trial_sequence)
        history_minimum_cost.append(minimum_cost)

    return history_minimum_cost, history_cost, sequence_history, optimal_sequence_history
# Gradient decent 
def gd(evaluate_cost, compute_gradient, initial_position, learning_rate, steps):
    position = np.array(initial_position, dtype=float)
    cost_record = []
    position_history = [position.copy()]
    for _ in range(steps):
        gradient = compute_gradient(*position)
        position -= learning_rate * gradient
        cost_record.append(evaluate_cost(*position))
        position_history.append(position.copy())

    return cost_record, position_history
# Global population
def gs_population(evaluate_cost, dim, bounds, samples, keep_best, iterations, epsilon):
    population = [np.array([uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]) for _ in range(samples)]
    cost_history = []
    parameter_history = []

    for _ in range(iterations):
        costs = [(evaluate_cost(*candidate), candidate) for candidate in population]
        sorted_population = sorted(costs, key=lambda x: x[0])
        best_candidates = sorted_population[:keep_best]

        cost_history.extend([cost for cost, _ in best_candidates])
        parameter_history.extend([candidate.tolist() for _, candidate in best_candidates])

        new_population = []
        for cost, best_candidate in best_candidates:
            for _ in range(samples // keep_best):
                neighbor = np.array([np.clip(uniform(best_candidate[i] - epsilon, best_candidate[i] + epsilon), bounds[i][0], bounds[i][1]) for i in range(dim)])
                new_population.append(neighbor)

        population = new_population

    return cost_history, parameter_history
# Stepwise plot
def analyze_and_visualize_stepwise(selected_function, search_count, gradient_steps, iterative_steps):
    function_list = [cost_function_one, cost_function_two]
    gradient_list = [gradient_one, gradient_two]
    range_limits = [(0, 15), (0, 15)]

    stepwise_random_history, random_evaluation_record, random_solution_history, optimal_random_solutions = gs_random(
        function_list[selected_function], 2, range_limits, search_count
    )

    stepwise_iterative_history, iterative_solution_history = gs_population(
        function_list[selected_function], 2, range_limits, search_count, keep_best=10, iterations=iterative_steps, epsilon=0.1
    )

    gradient_descent_history, descent_solution_history = gd(
        function_list[selected_function], gradient_list[selected_function], [3,3], 0.01, gradient_steps
    )

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(stepwise_random_history, 'b-', label='GS Random')
    plt.plot(stepwise_iterative_history, 'r-', label='GS Population')
    plt.plot(gradient_descent_history, 'g--', label='GD (α = 0.01)')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Cost Value')
    plt.title('Function Evaluations vs Cost Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(np.full_like(stepwise_random_history, 1/search_count)), stepwise_random_history, 'b-', label='GS Random')
    plt.plot(np.cumsum(np.full_like(stepwise_iterative_history, 1/(search_count*iterative_steps))), stepwise_iterative_history, 'r-', label='GS Population')
    plt.plot(np.cumsum(np.full_like(gradient_descent_history, 1/gradient_steps)), gradient_descent_history, 'g--', label='GD (α = 0.01)')
    plt.xlabel('Time (Arbitrary Units)')
    plt.ylabel('Cost Value')
    plt.title('Time vs Cost Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Contour plot
def analyze_and_visualize_contour(evaluate_func, calculate_gradient, sample_volume, descent_repetitions, population_cycles):
    constrained_bounds = [(5, 15), (3, 15)]

    x = np.linspace(0, 15, 400)
    y = np.linspace(0, 15, 400)
    X, Y = np.meshgrid(x, y)
    Z = evaluate_func(X, Y)

    plt.figure(figsize=(10, 8))
    contour_plot = plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35))
    plt.clabel(contour_plot, inline=1, fontsize=10)
    plt.colorbar(contour_plot)

    random_search_history, _, random_solution_paths, optimal_random_solutions = gs_random(
        evaluate_func, 2, [(0, 15), (0, 15)], sample_volume
    )
    
    iterative_search_history, iterative_solution_paths = gs_population(
        evaluate_func, 2, constrained_bounds, sample_volume, keep_best=10, iterations=population_cycles, epsilon=0.1
    )

    descent_evaluation_history, descent_path_history = gd(
        evaluate_func, calculate_gradient, [5, 5], 0.01, descent_repetitions
    )

    # Plotting paths
    x_vals_r, y_vals_r = zip(*optimal_random_solutions)
    plt.plot(x_vals_r, y_vals_r, 'o-', color='green', label='Global Random Search')

    x_vals_p, y_vals_p = zip(*iterative_solution_paths)
    plt.plot(x_vals_p, y_vals_p, 'o-', color='blue', label='Global Population Search Stepwise')

    x_vals_g, y_vals_g = zip(*descent_path_history)
    plt.plot(x_vals_g, y_vals_g, 'o-', color='red', label='Gradient Descent')

    # start and end points
    plt.annotate('Start', xy=(x_vals_r[0], y_vals_r[0]), xytext=(x_vals_r[0], y_vals_r[0]+1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('End', xy=(x_vals_r[-1], y_vals_r[-1]), xytext=(x_vals_r[-1], y_vals_r[-1]+1),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.annotate('Start', xy=(x_vals_p[0], y_vals_p[0]), xytext=(x_vals_p[0], y_vals_p[0]+1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('End', xy=(x_vals_p[-1], y_vals_p[-1]), xytext=(x_vals_p[-1], y_vals_p[-1]+1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.annotate('Start', xy=(x_vals_g[0], y_vals_g[0]), xytext=(x_vals_g[0], y_vals_g[0]+1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('End', xy=(x_vals_g[-1], y_vals_g[-1]), xytext=(x_vals_g[-1], y_vals_g[-1]+1),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend()
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('Contour plot showing the optimization paths')
    plt.show()

# Comparison for function 1
analyze_and_visualize_stepwise(0, 1000, 1000, 50)
analyze_and_visualize_contour(cost_function_one, gradient_one, 30, 300, 10)
# Comparison for function 2
analyze_and_visualize_stepwise(1, 550, 1000, 10)
analyze_and_visualize_contour(cost_function_two, gradient_two, 30, 700, 10)