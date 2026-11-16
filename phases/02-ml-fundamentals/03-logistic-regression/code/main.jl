# Main script for logistic regression demo
# This script demonstrates logistic regression and includes the demo_threshold_tuning function

using DelimitedFiles

# Define the logistic regression model type
struct LogisticRegression
    weights::Vector{Float64}
    bias::Float64
end

# Initialize model with random weights
function initialize_model(n_features)
    return LogisticRegression(randn(n_features), randn())
end

# Add intercept
function add_intercept(X)
    n_samples = size(X, 1)
    return vcat(X, ones(n_samples, 1))
end

# Sigmoid function
function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end

# Predict function
function predict(model, X)
    X_with_intercept = add_intercept(X)
    z = X_with_intercept * model.weights + model.bias
    return round.(sigmoid(z))
end

# Build metrics function with fix for empty collections
function build_metrics(y_true::Vector{Int64}, y_pred::Vector{Int64})
    # Fixed version with init=0 to handle empty generators
    tp = sum((y_true .== 1) & (y_pred .== 1), init=0)
    tn = sum((y_true .== 0) & (y_pred .== 0), init=0)
    fp = sum((y_true .== 0) & (y_pred .== 1), init=0)
    fn = sum((y_true .== 1) & (y_pred .== 0), init=0)
    
    # Calculate metrics with robust epsilon handling for all edge cases
    # Epsilon prevents division by zero while providing meaningful results
    eps = 1e-6
    
    # Accuracy: proportion of correct predictions (already safe with init=0)
    accuracy = sum(y_true .== y_pred, init=0) / length(y_true)
    
    # Precision: tp / (tp + fp) - safe from division by zero
    precision = tp / (tp + fp + eps)
    
    # Recall: tp / (tp + fn) - safe from division by zero
    recall = tp / (tp + fn + eps)
    
    # F1 Score:  정원 
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    
    return Dict(
        "accuracy" => accuracy,
        "precision" => precision,
        "recall" => recall,
        "f1_score" => f1_score,
        "true_positives" => tp,
        "true_negatives" => tn,
        "false_positives" => fp,
        "false_negatives" => fn
    )
end

# Demo function for threshold tuning
function demo_threshold_tuning(model::LogisticRegression, X_test::Vector{Vector{Float64}}, ys_test::Vector{Int64})
    println("Threshold Tuning Demo:")
    println("=========================")
    println("Testing various thresholds...")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds
        println("\nThreshold:", threshold)
        
        # Get predictions with current threshold
        y_pred = map(x -> sigmoid(x * model.weights + model.bias) >= threshold ? 1 : 0, X_test)
        
        # Build metrics
        metrics = build_metrics(ys_test, y_pred)
        
        println("Accuracy:", metrics["accuracy"])
        println("Precision:", metrics["precision"])
        println("Recall:", metrics["recall"])
        println("F1 Score:", metrics["f1_score"])
        println("TP:", metrics["true_positives"], " TN:", metrics["true_negatives"], " FP:", metrics["false_positives"], " FN:", metrics["false_negatives"])
    end
    
    println("\nThreshold tuning completed successfully!")
end

# Load data function
function load_data()
    # Create synthetic data for demo
    n_samples = 100
    n_features = 5
    
    # Generate random data
    X = [randn(n_features) for _ in 1:n_samples]
    ys = [rand() > 0.5 ? 1 : 0 for _ in 1:n_samples]
    
    return X, ys
end

# Main function
function main()
    println("Logistic Regression Demo")
    println("===========================")
    
    # Load data
    X, ys = load_data()
    
    # Split data
    n_train = div(length(X), 2)
    X_train = X[1:n_train]
    X_test = X[n_train+1:end]
    ys_train = ys[1:n_train]
    ys_test = ys[n_train+1:end]
    
    # Initialize model
    model = initialize_model(size(X_train[1])...)
    
    println("Model initialized with weights:", model.weights)
    println("Bias:", model.bias)
    
    # Demo threshold tuning
    demo_threshold_tuning(model, X_test, ys_test)
    
    println("\nDemo completed successfully!")
end

# Run the main function if this script is executed directly
if length(ARGS) == 0
    main()
end
