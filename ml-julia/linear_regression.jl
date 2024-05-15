using Random
using Plots
using LinearAlgebra
using Statistics

Random.seed!(123)

# Dataset
function generate_dataset()
    X = 2 * rand(500, 1)
    y = 5 .+ 3 .* X .+ randn(500, 1)
    return X, y
end

function plot_dataset(X, y)
    scatter(X, y, title="Dataset", xlabel="First feature", ylabel="Second feature")
end

function split_data(X, y, test_size=0.2)
    n_samples = size(X, 1)
    indices = shuffle(1:n_samples)
    split = floor(Int, test_size * n_samples)
    train_indices = indices[1:end-split]
    test_indices = indices[end-split+1:end]
    X_train, y_train = X[train_indices, :], y[train_indices, :]
    X_test, y_test = X[test_indices, :], y[test_indices, :]
    return X_train, X_test, y_train, y_test
end

# Linear regression functions
function initialize_parameters(n_features)
    weights = zeros(n_features, 1)
    bias = 0
    return weights, bias
end

function compute_cost(X, y, weights, bias)
    n_samples = size(X, 1)
    y_predict = X * weights .+ bias
    cost = (1 / n_samples) * sum((y_predict .- y) .^ 2)
    return cost
end

function compute_gradients(X, y, weights, bias)
    n_samples = size(X, 1)
    y_predict = X * weights .+ bias
    dJ_dw = (2 / n_samples) * X' * (y_predict .- y)
    dJ_db = (2 / n_samples) * sum(y_predict .- y)
    return dJ_dw, dJ_db
end

function update_parameters(weights, bias, dJ_dw, dJ_db, learning_rate)
    weights = weights .- learning_rate .* dJ_dw
    bias = bias .- learning_rate .* dJ_db
    return weights, bias
end

function train_gradient_descent(X, y; learning_rate=0.01, n_iters=100)
    n_samples, n_features = size(X)
    weights, bias = initialize_parameters(n_features)
    costs = []

    for i in 1:n_iters
        cost = compute_cost(X, y, weights, bias)
        push!(costs, cost)

        if i % 100 == 0
            println("Cost at iteration $i: $cost")
        end

        dJ_dw, dJ_db = compute_gradients(X, y, weights, bias)
        weights, bias = update_parameters(weights, bias, dJ_dw, dJ_db, learning_rate)
    end

    return weights, bias, costs
end

function train_normal_equation(X, y)
    weights = (X' * X) \ (X' * y)
    bias = 0
    return weights, bias
end

function predict(X, weights, bias)
    return X * weights .+ bias
end

function plot_cost(costs)
    plot(1:length(costs), costs, title="Development of cost during training",
         xlabel="Number of iterations", ylabel="Cost")
end

function compute_error(y_true, y_pred)
    n_samples = size(y_true, 1)
    error = (1 / n_samples) * sum((y_pred .- y_true) .^ 2)
    return error
end

function plot_predictions(X_train, y_train, X_test, y_p_test)
    scatter(X_train, y_train, label="Training data", color="blue")
    scatter!(X_test, y_p_test, label="Predictions", color="orange")
    xlabel!("First feature")
    ylabel!("Second feature")
    title!("Dataset in blue, predictions for test set in orange")
end

# Main program
X, y = generate_dataset()
plot_dataset(X, y)
X_train, X_test, y_train, y_test = split_data(X, y)

# Training with gradient descent
w_trained, b_trained, costs = train_gradient_descent(X_train, y_train, learning_rate=0.005, n_iters=600)
plot_cost(costs)

# Testing (gradient descent model)
n_samples, _ = size(X_train)
n_samples_test, _ = size(X_test)

y_p_train = predict(X_train, w_trained, b_trained)
y_p_test = predict(X_test, w_trained, b_trained)

error_train = compute_error(y_train, y_p_train)
error_test = compute_error(y_test, y_p_test)

println("Error on training set: $(round(error_train, digits=4))")
println("Error on test set: $(round(error_test, digits=4))")

# Training with normal equation
X_b_train = hcat(ones(n_samples, 1), X_train)
X_b_test = hcat(ones(n_samples_test, 1), X_test)

w_trained_normal = train_normal_equation(X_b_train, y_train)

# Testing (normal equation model)
y_p_test_normal = predict(X_b_test, w_trained_normal[1], w_trained_normal[2])

error_train_normal = compute_error(y_train, y_p_train)
error_test_normal = compute_error(y_test, y_p_test_normal)

println("Error on training set (normal equation): $(round(error_train_normal, digits=4))")
println("Error on test set (normal equation): $(round(error_test_normal, digits=4))")

# Visualize test predictions
plot_predictions(X_train, y_train, X_test, y_p_test)
