## Instructions
```shell
. envar.sh
```
cd to `./scripts`

Evaluation:
```shell
python evaluation.py \
configs/HighwayEnv/env_continuous_actions.json \
configs/HighwayEnv/agents/sb3/ppo.json \
out/highway_env_continuous_actions/stable_baselines3.ppo.ppo/2024-10-08_23_18_16/best_model.zip \
--test
```


Inference:
```shell
python inference.py \
configs/HighwayEnv/env_continuous_actions.json \
configs/HighwayEnv/agents/sb3/ppo.json \
out/highway_env_continuous_actions/stable_baselines3.ppo.ppo/2024-10-08_23_18_16/best_model.zip \
--test
```