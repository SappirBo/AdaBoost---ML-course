'''
AdaBoost implementation for Machine Learning course - Ariel university

Main page to run the code and test the algorithm with different configurations 

Authors: Sappir Bohbot & Almog David
'''

import AdaBoost

ada_boost = AdaBoost.AdaBoost()

# Setting the run configursations here: 
runs_of_adaboost = 1            # Number of time you run the AdaBoost algorithm.
line_or_circle   = 1            # 1- lines. 2 - circles.
iterations_each_run = 8         # Number of iterations each time you run the AdaBoost algorithm.
print_iteration_details = True  # True  - print the details. 
                                # False - keep the print clean (Only print in the end the average empirical and true error).  
to_visualize = False            # set True to show visualized diagram.


    
empirical_error_sum = 0
true_error_sum = 0

for i in range(0, runs_of_adaboost):
    accuracy_S, accuracy_T = ada_boost.run(line_or_circle, iterations_each_run, print_iteration_details, True )

    empirical_error_sum += ( accuracy_S)
    true_error_sum += ( accuracy_T)
    

empirical_err = empirical_error_sum / runs_of_adaboost
true_err = true_error_sum / runs_of_adaboost

print(f"avarge empirical error: {empirical_err}")
print(f"avarge true error     : {true_err}")


