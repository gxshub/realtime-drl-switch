#!/bin/bash

# Make the script executable before running it
# chmod +x <script_name.sh>

# This shell scripts runs 'experiment.py'
# to evaluate the agent's performance
# in the presence of with varying time delays.

LD_LIBRARY_PATH=$(cd .. && pwd)/venv/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.05

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.1

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.15

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.2

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.25

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.3

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.35

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.4

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.45

python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-p -e 500 -o run1 \
-d 0.5