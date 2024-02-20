'''
AdaBoost implementation for Machine Learning course - Ariel university

This is helper for the adaboost -> Containing some tools and functions to make the AdaBoost class more
readable and more flexable

Authors: Sappir Bohbot & Almog David
'''

import random
from itertools import combinations



class AdaBoostTools:
    def __init__(self) -> None:
        pass

    def get_points(self, target: str)-> list:
        # Define a list to hold the points
        points = []

        # Open the file and read line by line
        with open(target, 'r') as file:
            for line in file:
                # Split each line into components: x, y, and label
                x_str, y_str, label_str = line.split()
                
                # Convert x, y to float and label to int
                x = float(x_str)
                y = float(y_str)
                label = int(label_str)
                
                # Append the tuple (x, y, label) to the points list
                points.append((x, y, label))

        return points
    
    def split_to_t_and_s(self, points: list)->tuple:
        # Shuffle the points list in place
        random.shuffle(points)

        # Define the split size => 50% for S, 50% for T)
        split_size = int(len(points) * 0.5)

        # Split the points into S and T
        S = points[:split_size]
        T = points[split_size:]

        return S,T

    def get_initial_weight(self, set_of_points: list) -> dict:
        size = len(set_of_points)  # Calculate the size of set_of_points
        initial_value = 1 / size  # Calculate the initial value for each point

        # Create the dictionary with initial values
        return {tuple(point): initial_value for point in set_of_points}

    def get_set_of_hypothesis(self, set_of_points: list):
        return list(combinations(set_of_points, 2))

    def evaluate_hypothesis(self, hypothesis, points_weight, points, line_or_circle):
        error = 0
        predictions = {}
        for point in points:
            predicted_label = self.predict(hypothesis, point, line_or_circle)
            actual_label = point[2]  # label is the third element in the tuple
            predictions[point] = predicted_label
            
            if predicted_label != actual_label:
                error += points_weight[point]
        
        return error, predictions

    def predict(self, hypothesis, point, line_or_circle):
        A, B = hypothesis  # Unpack the two points defining the hypothesis
        P = point  # The point to classify
        # Extract coordinates
        Ax, Ay = A[0], A[1]
        Bx, By = B[0], B[1]
        Px, Py = P[0], P[1]
        
        if line_or_circle == 1: # Line hypothesis
            # Calculate the determinant (cross product in 2D) to determine the side
            determinant = (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)
            # Return +1 or -1 based on the side of the line
            return 1 if determinant > 0 else -1
        
        elif line_or_circle == 2: # Circle hypothesis
            # Calculate the radius of the circle (distance between A and B)
            radius = ((Bx - Ax)**2 + (By - Ay)**2)**0.5
            # Calculate the distance from P to A (center of the circle)
            distance_PA = ((Px - Ax)**2 + (Py - Ay)**2)**0.5
            # If the distance from P to A is less than or equal to the radius, P is inside the circle
            return 1 if distance_PA <= radius else -1

    def predict_test_data(self, T, best_hypotheses, alphas, line_or_circle):
        final_predictions = []
        for point in T:
            # Calculate the weighted sum of predictions from all hypotheses
            weighted_sum = sum(alpha * self.predict(hypothesis, point, line_or_circle) for hypothesis, alpha in zip(best_hypotheses, alphas))
            
            # The final prediction is based on the sign of the weighted sum
            final_prediction = 1 if weighted_sum > 0 else -1
            final_predictions.append((point, final_prediction))
        
        return final_predictions

    def calculate_accuracy(self, predictions):
        correct = 0
        for point, predicted_label in predictions:
            actual_label = point[2]  # Assuming the label is the third element in the tuple
            if predicted_label == actual_label:
                correct += 1
        accuracy = correct / len(predictions)
        return accuracy





