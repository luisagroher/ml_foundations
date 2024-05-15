using Random
using Plots
using LinearAlgebra
using Statistics

Random.seed!(123)

# Dataset
function generate_dataset(n_samples=1000, centers=2, cluster_std=1.0)
    # Generate random centers
    center_X_1 = rand(2) * 5
    center_X_2 = rand(2) * 5

    # Generate random samples around the centers
    X_1 = randn(n_samples ÷ 2, 2) .* cluster_std .+ center_X_1'
    X_2 = randn(n_samples ÷ 2, 2) .* cluster_std .+ center_X_2'

    # Combine the samples
    X = vcat(X_1, X_2)

    # Create the corresponding labels
    y_true = vcat(zeros(n_samples ÷ 2), ones(n_samples ÷ 2))

    # Plot the dataset
    colors = [y == 1 ? :red : :blue for y in y_true]
    scatter(X[:, 1], X[:, 2], c=colors, title="Dataset", xlabel="First feature", ylabel="Second feature")

    return X, y_true
end

function preprocess_data(X, y_true, test_size=0.2)
    y_true = reshape(y_true, :, 1)
    n_samples = size(X, 1)

    # Generate random indices for splitting the data
    indices = shuffle(1:n_samples)
    split_idx = floor(Int, test_size * n_samples)

    # Split the data into training and testing sets
    X_train = X[indices[1:end-split_idx], :]
    y_train = y_true[indices[1:end-split_idx], :]
    X_test = X[indices[end-split_idx+1:end], :]
    y_test = y_true[indices[end-split_idx+1:end], :]

    println("Shape X_train: $(size(X_train))")
    println("Shape y_train: $(size(y_train))")
    println("Shape X_test: $(size(X_test))")
    println("Shape y_test: $(size(y_test))")

    return X_train, X_test, y_train, y_test
end

# Logistic regression functions
sigmoid(a) = 1 / (1 + exp(-a))

function initialize_parameters(n_features)
    weights = zeros(n_features, 1)
    bias = 0
    return weights, bias
end

function compute_cost(X, y_true, weights, bias)
    n_samples = size(X, 1)
    y_predict = reshape(sigmoid.(X * weights .+ bias), n_samples)
    cost = (- 1 / n_samples) * sum(y_true .* log.(y_predict) .+ (1 .- y_true) .* log.(1 .- y_predict))
    return cost
end

function compute_gradients(X, y_true, y_predict)
    n_samples = size(X, 1)
    y_true = reshape(y_true, n_samples)
    y_predict = reshape(y_predict, n_samples)
    dw = (1 / n_samples) * X' * (y_predict - y_true)
    db = (1 / n_samples) * sum(y_predict - y_true)
    return dw, db
end

function update_parameters(weights, bias, dw, db, learning_rate)
    weights -= learning_rate .* dw
    bias -= learning_rate * db
    return weights, bias
end

function train(X, y_true; n_iters=100, learning_rate=0.01)
    n_samples, n_features = size(X)
    weights, bias = initialize_parameters(n_features)
    costs = []

    for i in 1:n_iters
        y_predict = reshape(sigmoid.(X * weights .+ bias), n_samples)
        cost = compute_cost(X, y_true, weights, bias)
        dw, db = compute_gradients(X, y_true, y_predict)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)

        push!(costs, cost)
        if i % 100 == 0
            println("Cost after iteration $i: $cost")
        end
    end

    return weights, bias, costs
end

function predict(X, weights, bias)
    y_predict = sigmoid.(X * weights .+ bias)
    y_predict_labels = [elem > 0.5 ? 1 : 0 for elem in y_predict]
    return reshape(y_predict_labels, :, 1)
end

function plot_cost(costs)
    plot(1:length(costs), costs, title="Development of cost over training",
         xlabel="Number of iterations", ylabel="Cost")
end

function calculate_accuracy(y_true, y_pred)
    return 100 - mean(abs.(y_pred - y_true)) * 100
end

# Main program
X, y_true = generate_dataset()
X_train, X_test, y_train, y_test = preprocess_data(X, y_true)

w_trained, b_trained, costs = train(X_train, y_train, n_iters=600, learning_rate=0.009)
plot_cost(costs)

y_p_train = predict(X_train, w_trained, b_trained)
y_p_test = predict(X_test, w_trained, b_trained)

train_accuracy = calculate_accuracy(y_train, y_p_train)
test_accuracy = calculate_accuracy(y_test, y_p_test)

println("train accuracy: $train_accuracy%")
println("test accuracy: $test_accuracy%")