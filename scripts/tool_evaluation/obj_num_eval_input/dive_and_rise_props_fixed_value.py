import math
import sys
import random

def truncate_float(number, decimal_places):
    factor = 10**decimal_places
    return math.floor(number * factor) / factor

# Check the number of command-line arguments
num_args = len(sys.argv)

file_name_prefix = sys.argv[1]        # The first argument is the filename
n_targets = int(sys.argv[2])    # Number of targets

value = truncate_float(0.25/n_targets, 3)   # Lower-bound value
#value = float(sys.argv[3])

y_max = 99  # must be the same as defined in the model file
inc_index = math.floor(y_max / (n_targets - 1))

file_name = file_name_prefix+'_'+str(n_targets)+'.props'
with open(file_name, 'w') as file:
    file.write("multi(\n")
    for i in range(n_targets-1):
        j = i * inc_index
        file.write(f'R{{\"target_{j}\"}}>={value} [ C ],\n')
    j = (n_targets - 1) * inc_index
    file.write(f'R{{\"target_{j}\"}}>={value} [ C ]\n')
    file.write(")")

print(f"wrote property to file: {file_name}")