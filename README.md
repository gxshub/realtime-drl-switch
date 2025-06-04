# Instructions

Navigate to the folder `./scripts`.
Set `LD_LIBRARY_PATH` environment variable:
```shell
export LD_LIBRARY_PATH=$(cd .. && pwd)/venv/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

Train an agent:
```shell
python training.py \
configs/HighwayEnv/env_continuous_v2.json \
configs/HighwayEnv/agents/sb3/ddpg_2.json \
--total-timesteps=1000 \
--processes=2 \
--trial-mode
```

Evaluate a trained agent:
```shell
python evaluation.py \
configs/HighwayEnv/env_continuous_v2.json \
configs/HighwayEnv/agents/sb3/ddpg_2.json \
out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
--test
```

<!--
For real-time inference:
```shell
python inference.py \
configs/HighwayEnv/env_continuous_actions.json \
configs/HighwayEnv/agents/sb3/ddpg.json \
out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
--test
```

For controller-only evaluation:
```shell
python evaluation_co.py \
configs/HighwayEnv/env_continuous_actions.json \
--test
```
-->

Run experiments:
```shell
python experiment.py \
configs/HighwayEnv/env_continuous_v2.json \
-w configs/HighwayEnv/switch/switch_dc0.yaml \
-a configs/HighwayEnv/agents/sb3/ddpg.json \
-c out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
-s configs/HighwayEnv/controller/ttc_based.yaml \
-t
```

To disable depreciation warnings by Gymnasium in the console, run the following command before the evaluation or inference. 
```shell
 export PYTHONWARNINGS=ignore
```
