#!/bin/bash

# Make the script executable before running it
# chmod +x <script_name.sh>

# This shell scripts runs 'experiment.py'
# with different configurations.
# The resulted date is used to determine
# probability parameters in the switch model.

LD_LIBRARY_PATH=$(cd .. && pwd)/venv/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-w configs/HighwayEnv/switch/switch_dc0.yaml \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-e 1000

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-w configs/HighwayEnv/switch/switch_dc1.yaml \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-e 1000

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-w configs/HighwayEnv/switch/switch_dc2.yaml \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-e 1000

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-w configs/HighwayEnv/switch/switch_dc3.yaml \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-e 1000

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-p \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-e 1000