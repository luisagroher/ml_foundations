using Random
using Statistics
using LinearAlgebra
using Plots
using Distributions
using MLJ
using Tables
using CategoricalArrays

#function make_blobs(n_samples::Int=1000, centers::Int=2, cluster_std::Float64=1.0)
#    center_X_1 = rand(2) * 5
#    center_X_2 = rand(2) * 5
#
#    X_1 = randn(n_samples ÷ 2, 2) .* cluster_std .+ center_X_1'
#    X_2 = randn(n_samples ÷ 2, 2) .* cluster_std .+ center_X_2'
#    
#    X = vcat(X_1, X_2)
#
#    y_true = vcat(zeros(n_samples ÷ 2 ), ones(n_samples ÷ 2))
#    return X, y_true
#end


function transform_targets(targets)
    return map(t -> t == 1 ? -1 : +1, targets)
end


function plot_dataset(features, targets)
    scatter(features[1, :], features[2, :], c=targets, title="Toy dataset",
            xlabel="Feature 1", ylabel="Feature 2", legend=false)
end

function add_bias_term(features)
    d = size(features)
    n_samples = size(features, 1)
    return hcat(ones(n_samples, 1), features)
end

function compute_cost(features, labels, weights, regularization_param)
    n_samples = size(features, 1)

    predictions = features * weights
    #predictions = dot.(eachcol(features), Ref(weights))
    distances = 1 .- labels .* predictions
    hinge_losses = max.(0, distances)
    sum_hinge_loss = sum(hinge_losses) / n_samples
    cost = (1 / 2) * dot(weights, weights) + regularization_param * sum_hinge_loss
    return cost
end

function compute_gradient(features, labels, weights, regularization_param)
    # predictions = dot.(eachcol(features), Ref(weights))
    predictions = features * weights
    distances = 1 .- labels .* predictions
    n_samples, n_feat = size(features)
    sub_gradients = zeros(1, n_feat)

    for (idx, dist) in enumerate(distances)
        if max(0, dist) == 0
            sub_gradients .+= weights'
        else
            sub_grad = weights' .- (regularization_param .* features[:, idx] .* labels[idx])
            sub_gradients .+= sub_grad
        end
    end
                
    avg_gradient = sum(sub_gradients) ./ length(labels)
    return avg_gradient
end

function train_svm(train_features, train_labels; n_epochs=10, learning_rate=0.05, batch_size=1, regularization_param=100)
    train_features = add_bias_term(train_features)
    n_samples, n_feat = size(train_features)
    weights = zeros(n_feat, 1)
    
    for epoch in 1:n_epochs
        features, labels = train_features, train_labels
        start, stop = 1, batch_size
        while stop <= length(labels)
            batch = features[start:stop,:]
            batch_labels = labels[start:stop]
            
            grad = compute_gradient(batch, batch_labels, weights, regularization_param)
            update = (learning_rate .* grad)'
            weights .-= update
            start, stop = stop + 1, stop + batch_size
        end
        current_cost = compute_cost(features, labels, weights, regularization_param)
        println("Epoch $epoch, cost: $current_cost")
    end
    
    return weights
end

function predict(test_features, trained_weights)
    test_features = add_bias_term(test_features)
    if trained_weights === nothing
        error("You haven't trained the SVM yet!")
    end
    predicted_labels = trunc.(Int, sign.(test_features * trained_weights))
    
    # predicted_labels = sign.(dot.(eachcol(test_features), Ref(trained_weights)))
    return predicted_labels
end


function visualize_decision_boundary(data_features, data_targets, weights)
    # Create a scatter plot for the two clusters
    scatter(data_features[:, 1], data_features[:, 2], group=data_targets, legend=:outerbottomright, markersize=3, alpha=0.5)
    x1_range = range(minimum(data_features[:, 1]), maximum(data_features[:, 1]), length=100)
    x2_range = range(minimum(data_features[:, 2]), maximum(data_features[:, 2]), length=100)
    # x1_grid, x2_grid = meshgrid(x1_range, x2_range)
    x1_grid = repeat(x1_range, outer=length(x2_range))
    x2_grid = repeat(x2_range, inner=length(x1_range))
    
    # Evaluate the decision function on the grid
    decision_grid = weights[1] * x1_grid + weights[2] * x2_grid
    
    # Plot the decision boundary
    contour!(x1_range, x2_range, decision_grid, levels=[0], color=:black, linewidth=2)
end

function preprocess_data(X, y_true; test_size=0.2)
    y_true = reshape(y_true, :, 1)
    n_samples = size(X, 1)

    indices = shuffle(1:n_samples)
    split_idx = floor(Int, test_size * n_samples)

    X_train = X[indices[1:end-split_idx], :]
    y_train = y_true[indices[1:end-split_idx], :]
    X_test = X[indices[end-split_idx+1:end], :]
    y_test = y_true[indices[end-split_idx+1:end], :]
    println("Shape X_train: $(size(X_train))")
    println("Shape y_train: $(size(y_train))")
    println("Shape X_test: $(size(X_test))")
    println("Shape y_test: $(size(y_test))")
    
    return X_train, y_train, X_test, y_test
end


function train_test_svm()
    # data_features, data_targets = make_blobs(600, 2, 2, random_state=42)
    #data_features, data_targets = make_blobs(1000, 2, 1.0)
    data_features, data_targets = make_blobs(600, 2; centers=2, rng=42)
    data_features, data_targets = Tables.matrix(data_features), levelcode.(data_targets)
    transformed_data_targets = transform_targets(data_targets)

    features_train, labels_train, features_test, labels_test = preprocess_data(data_features, transformed_data_targets, test_size=0.3)

    regularization_param = 100
    lr = 0.000001
    trained_weights = train_svm(features_train, labels_train, n_epochs=10, learning_rate=lr, batch_size=1, regularization_param=regularization_param)


    svm_predict(features) = predict(features, trained_weights)
    predicted_labels = svm_predict(features_test)

    accuracy = mean(predicted_labels .== labels_test)
    println("Accuracy on test dataset: $accuracy")
   
    visualize_decision_boundary(data_features, transformed_data_targets, trained_weights)
    
end

train_test_svm()