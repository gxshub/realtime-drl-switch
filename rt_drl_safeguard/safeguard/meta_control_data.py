# TTC-based controller
CTRL_PARAMS = {
    'MAX_SPEED': 30.0,  # m/s
    'MIN_SPEED': 0.0,  # m/s
    'DELTA_SPEED': 5,  # m/s
    'ACC_DELTA_MULTIPLIERS': [0.1, 0.05, 0, -0.5, -1.0, -2.0, -3.0],
    # e.g., 0.1 indicates a speed change of +0.1*DELTA_SPEED
    # -0.5 indicates a speed change of -0.5*DELTA_SPEED
    'ACC_TIME_THD': 5.5,  # s
    'SW_TIME_THD': 5.5, # s
    'SW_DELTA_SPEED': 6 # m/s
}

# Plan-based controller
MAX_SPEED = 30.0  # m/s
MIN_SPEED = 0.0  # m/s
DELTA_SPEED = 5  # m/s
DEFAULT_HORIZON = 10
DEFAULT_TIME_QUANTIZATION = 1
SPEED_LEVELS = 11 # an odd number > 2