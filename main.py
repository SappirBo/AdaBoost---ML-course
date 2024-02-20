'''
AdaBoost implementation for Machine Learning course - Ariel university

Main page to run the code and test the algorithm with different configurations 

Authors: Sappir Bohbot & Almog David
'''

import AdaBoost

ada_boost = AdaBoost.AdaBoost()

runs = 1

empirical_error_sum = 0
true_error_sum = 0

for i in range(0,runs):
    accuracy_S, accuracy_T = ada_boost.run(1, True, False )

    empirical_error_sum += ( accuracy_S)
    true_error_sum += ( accuracy_T)
    

empirical_err = empirical_error_sum/runs
true_err = true_error_sum/runs

print(f"avarge empirical error: {empirical_err}")
print(f"avarge true error     : {true_err}")


