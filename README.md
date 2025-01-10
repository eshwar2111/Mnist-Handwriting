Import Libraries: Essential libraries for building and training the model.
Load Dataset: Use tf.keras.datasets to load MNIST data, splitting it into training and testing sets.
Preprocess Data: Normalize pixel values and reshape for CNN compatibility.
Build Model: A basic CNN with one convolutional layer, pooling, and dense layers for classification.
Compile Model: Use the adam optimizer and categorical cross-entropy loss for multi-class classification.
Train Model: Train for 3 epochs to save time while achieving good performance.
Evaluate Model: Print test accuracy to evaluate model performance.
Make Predictions: Use the trained model to predict classes for a few test images.
Visualize Results: Display test images with their true labels and model predictions.

Training Accuracy: ~92%
Test Accuracy: ~91%
Visualization: Images with predicted and true labels displayed.
