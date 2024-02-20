'''
AdaBoost implementation for Machine Learning course - Ariel university

AdaBoost class - create an AdaBoost element that runs the algorithm with printing and visualize options.

Authors: Sappir Bohbot & Almog David
'''
import numpy as np
import matplotlib.pyplot as plt
import AdaBoostTools

class AdaBoost:
    def __init__(self) -> None:
        pass
    
    '''
    Run the adaboost according to Hypothses -> lines or circles (1 for lines, 2 for circles).
    '''
    def run(self, line_or_circle:int,iterations_number:int, print_details = False, to_visualize = False): 
        if line_or_circle != 1 and line_or_circle != 2:
            print("Eror")
            exit
        else:
            pass

        iterations = iterations_number # the so called "k" in the class algorithm

        # Tools is helper to save the functions in order
        tools = AdaBoostTools.AdaBoostTools()

        # Read all points in circle_separator.txt - each point will be a tuple 
        points = tools.get_points('circle_separator.txt')

        # S for training data and T for test data
        S,T = tools.split_to_t_and_s(points)

        # Initilize points weights for S - Create the dictionary with initial values
        points_weight = tools.get_initial_weight(S)

        # Creating set of hypotheses - every two points of S is hypotheses.
        H = tools.get_set_of_hypothesis(S)

        best_hypotheses = []
        alphas = []

        for i in range(iterations):
            best_hypothesis = None
            best_error = float('inf')
            best_predictions = None
            
            for hypothesis in H:
                error, predictions = tools.evaluate_hypothesis(hypothesis, points_weight, S, line_or_circle)
                
                if error < best_error:
                    best_error = error
                    best_hypothesis = hypothesis
                    best_predictions = predictions
            
            # Calculate alpha
            alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
            alphas.append(alpha)
            best_hypotheses.append(best_hypothesis)
            
            # Update weights
            for point in S:
                points_weight[point] *= np.exp(-alpha * best_predictions[point] * point[2])
            Z = sum(points_weight.values())  # Normalization factor
            points_weight = {point: weight / Z for point, weight in points_weight.items()}

            # Predict the labels of the test data S
            test_predictions_s = tools.predict_test_data(S, best_hypotheses, alphas, line_or_circle)
            # Calculate the accuracy of the predictions S
            accuracy_S = tools.calculate_accuracy(test_predictions_s)
            empirical_error_s = 1 - accuracy_S
            
            if print_details:
                print(f'Iteration {i}: ')
                print(f"        - empirical_error (on set S): {empirical_error_s * 100:.2f}%")
                # print(f"        - True error on T: {accuracy_T * 100:.2f}%")


        if len(alphas) != len(best_hypotheses):
            # Error !!
            if print_details:
                print("Eror: length of alphas is not equals to the length of best_hypotheses")
        else:
            if print_details:
                for i in range(0,len(alphas)):
                    print(f"{i}. Hypothese {i+1}: {best_hypotheses[i]},\n   -- -- -- Alpha: {alphas[i]}")

        # Predict the labels of the test data for S
        test_predictions_s = tools.predict_test_data(S, best_hypotheses, alphas, line_or_circle)
        # Calculate the accuracy of the predictions for S
        accuracy_S = tools.calculate_accuracy(test_predictions_s)

        # Predict the labels of the test data for T
        test_predictions_t = tools.predict_test_data(T, best_hypotheses, alphas, line_or_circle)
        # Calculate the accuracy of the predictions for T
        accuracy_T = tools.calculate_accuracy(test_predictions_t)

        empirical_error_s = 1 - accuracy_S
        true_error_t = 1 - accuracy_T

        if print_details:
            print(f"Empirical error on S: {empirical_error_s * 100:.2f}%")
            print(f"True error on T: {true_error_t * 100:.2f}%")

        if to_visualize:
            self.visualize(points, best_hypotheses, line_or_circle)
        
        return empirical_error_s, true_error_t
        pass

    def visualize(self, S, best_hypotheses, line_or_circle):
        # Plot all points
        for point in S:
            plt.scatter(point[0], point[1], color='blue' if point[2] == -1 else 'red')

        # Set the limits of the plot to the new range
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

        if line_or_circle == 1:
            # Plot each line
            x_values = np.linspace(-2.5, 2.5, 400)
            for hypothesis in best_hypotheses:
                A, B = hypothesis # Assuming hypothesis structure is ((pointA, pointB), line_or_circle)
                if A[0] == B[0]:  # Vertical line
                    plt.axvline(x=A[0])
                else:
                    slope = (B[1] - A[1]) / (B[0] - A[0])
                    intercept = A[1] - slope * A[0]
                    y_values = slope * x_values + intercept
                    plt.plot(x_values, y_values, '-r')
        elif line_or_circle == 2:
            # Plot each circle
            for hypothesis in best_hypotheses:
                A, B = hypothesis
                radius = np.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)
                circle = plt.Circle((A[0], A[1]), radius, color='r', fill=False)
                plt.gca().add_artist(circle)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('AdaBoost Classification')
        plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
        plt.show()


    