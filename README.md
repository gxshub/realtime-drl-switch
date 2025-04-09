## Running Instructions

Set `LD_LIBRARY_PATH` environment variable:
```shell
export LD_LIBRARY_PATH=$(pwd)/venv/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```
Navigate to the folder `./scripts`

For evaluation:
```shell
python evaluation.py \
configs/HighwayEnv/env_continuous_actions.json \
configs/HighwayEnv/agents/sb3/ddpg.json \
out/highway_env_continuous_actions/stable_baselines3.ddpg.ddpg/2025-02-12_18_16_34/best_model.zip \
--test
```

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

To disable depreciation warnings by Gymnasium in the console, run the following command before the evaluation or inference. 
```shell
 export PYTHONWARNINGS=ignore
```

---
```shell
export agent_config= # e.g. td3.json
```
```shell
export file_path= #e.g. out/highway_env_continuous_actions/stable_baselines3.td3.tds/2025-01-22_10_22_22/best_model.zip
```
```shell
python evaluation.py configs/HighwayEnv/env_continuous_actions.json configs/HighwayEnv/agents/sb3/$agent $file_path --episodes=1000 --processes=6
```