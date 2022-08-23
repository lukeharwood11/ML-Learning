import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    """
    :resource: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    :return:
    """
    # Read in data and display the first 5 rows
    path = 'res/temps.csv'
    df = pd.read_csv(path)
    print(df.head())

    print('The shape of our features is:', df.shape)

    # To identify anomalies, we can quickly compute summary statistics.
    print(df.describe())

    # One-Hot encode the data
    features = pd.get_dummies(df)

    # Display the first 5 rows of the last 12 columns
    print(features.iloc[:, 5:].head(5))

    labels = np.array(features['actual'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('actual', axis=1)

    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)

    # Order:
    # One-hot encoded categorical variables
    # Split data into features and labels
    # Converted to arrays
    # Split data into training and testing sets

    # Establishing a baseline is important, so we have a basis for whether or not this solution is viable/worth it
    baseline_preds = X_test[:, feature_list.index('average')]

    # Baseline errors, and display the average baseline error
    baseline_errors = abs(baseline_preds - y_test)

    print('Average baseline error:', round(np.mean(baseline_errors), 2))

    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on the training data
    rf.fit(X_train, y_train)

    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)

    # Calculate the absolute errors
    errors = abs(predictions - y_test)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors/y_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    # Visualization
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest

    tree = rf.estimators_[5]

    # Export the image to a dot file
    export_graphviz(tree, out_file='res/tree.dot', feature_names=feature_list, rounded=True, precision=1)

    # Use dot file to create a graph
    (graph,) = pydot.graph_from_dot_file('res/tree.dot')

    # Write graph to a png file
    graph.write_png('res/tree.png')

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # Look at the correlation between different data points
    print(df.corr()['actual'])


if __name__ == "__main__":
    main()
