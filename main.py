'''
AdaBoost implementation for Machine Learning course - Ariel university

Main page to run the code and test the algorithm with different configurations 

Authors: Sappir Bohbot & Almog David
'''

import AdaBoost

ada_boost = AdaBoost.AdaBoost()

# Setting the run configursations here: 
runs_of_adaboost = 50            # Number of time you run the AdaBoost algorithm.
line_or_circle   = 1            # 1- lines. 2 - circles.
iterations_each_run = 8         # Number of iterations each time you run the AdaBoost algorithm.
print_iteration_details = False  # True  - print the details. 
                                # False - keep the print clean (Only print in the end the average empirical and true error).  
to_visualize = False            # set True to show visualized diagram.
show_best_hypotheses = False    # in each iteration print best_hypotheses


# Setting up all the Information
empirical_error_sum = 0
true_error_sum = 0
empirical_errors_ave_per_iteration = {}
true_error_ave_per_iteration = {}

for iter  in range(0,iterations_each_run):
    empirical_errors_ave_per_iteration[iter] = 0
    true_error_ave_per_iteration[iter] = 0

'''
Main loop!
Here we do the "ada_boost.run" function that returns to as the sum of all the Iterations ands their summery.
'''
for i in range(0, runs_of_adaboost):
    empirical_error_s, true_error_t, empirical_errors_set, true_error_set = ada_boost.run(line_or_circle, iterations_each_run, print_iteration_details, 
                                           to_visualize, show_best_hypotheses)

    empirical_error_sum += ( empirical_error_s)
    true_error_sum += ( true_error_t)

    for iter  in range(0,iterations_each_run):
        empirical_errors_ave_per_iteration[iter] += empirical_errors_set[iter]
        true_error_ave_per_iteration[iter]       += true_error_set[iter]


# Analysis of the returned data
for iter  in range(0,iterations_each_run):
        empirical_errors_ave_per_iteration[iter] /= runs_of_adaboost
        true_error_ave_per_iteration[iter]       /= runs_of_adaboost

# Printing the results    
if runs_of_adaboost > 0:
    empirical_err = empirical_error_sum / runs_of_adaboost
    true_err = true_error_sum / runs_of_adaboost

    for iter  in range(0,iterations_each_run):
        print(f'Average error in iteration {iter}:')
        print(f'    - empirical error: {empirical_errors_ave_per_iteration[iter]}') 
        print(f'    - true error     : {true_error_ave_per_iteration[iter]      }')  

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(f"Avarge Empirical Error: {empirical_err}")
    print(f"Avarge True Error     : {true_err}")

# End
print("\n Out ...")