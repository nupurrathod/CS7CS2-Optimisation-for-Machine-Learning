import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import uniform
from tensorflow import keras
from keras import layers, regularizers
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

def loss(batch_size, alpha, beta1, beta2, epochs):
    batch_size = int(batch_size)
    epochs = int(epochs)
    num_classes = 10
    input_shape = (32, 32, 3)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n = 5000
    x_train = x_train[:n]; y_train = y_train[:n]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001))
    ])

    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)

    y_preds = model.predict(x_test)
    loss_fn = CategoricalCrossentropy()
    return loss_fn(y_test, y_preds).numpy()

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
        for _, best_candidate in best_candidates:
            for _ in range(samples // keep_best):
                neighbor = np.array([np.clip(best_candidate[i] + uniform(-epsilon, epsilon), bounds[i][0], bounds[i][1]) for i in range(dim)])
                new_population.append(neighbor)
        population = new_population

    return cost_history, parameter_history


n = 5
range_minibatch = [[1, 130], [0.01, 0.1], [0.9, 0.99], [0.999, 0.9999], [20, 20]]
gs_cost_minibatch, _, _, gs_param_minibatch = gs_random(loss, n, range_minibatch, 50)
gp_cost_minibatch, gp_param_minibatch = gs_population(loss, n, range_minibatch, 12, 4, 4, 0.1)

# Param optimization
range_param = [[38, 38], [0.0001, 0.1], [0.25, 0.99], [0.9, 0.9999], [20, 20]]
gs_cost_param, _, _, gs_param_param = gs_random(loss, n, range_param, 50)
gp_cost_param, gp_param_param = gs_population(loss, n, range_param, 12, 4, 4, 0.1)

# Epochs optimization
range_epochs = [[38, 38], [0.01, 0.01], [0.9, 0.9], [0.999, 0.999], [10, 30]]
gs_cost_epochs, _, _, gs_param_epochs = gs_random(loss, n, range_epochs, 50)
gp_cost_epochs, gp_param_epochs = gs_population(loss, n, range_epochs, 12, 4, 4, 0.1)

# Output best batch sizes from minibatch optimization
best_batch_size_r = gs_param_minibatch[0]
best_batch_size_p = gp_param_minibatch[0]
print('Best batch size Random = ', best_batch_size_r)
print("Best batch size Population =", best_batch_size_p)

best_epoch_r = gs_param_epochs[0]
best_epoch_p = gp_param_epochs[0]
print('Best epochs Random = ',best_epoch_r)
print("Best epochs Population =", best_epoch_p)

def plot_results(gs_param, gs_cost, gp_param, gp_cost):
    plt.plot(range(len(gs_cost)), gs_cost, label='GS Random')
    plt.plot(range(len(gp_cost)), gp_cost, label='GS Population')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plotting results
plot_results(gs_param_minibatch, gs_cost_minibatch, gp_param_minibatch, gp_cost_minibatch)
plot_results(gs_param_param, gs_cost_param, gp_param_param, gp_cost_param)
plot_results(gs_param_epochs, gs_cost_epochs, gp_param_epochs, gp_cost_epochs)