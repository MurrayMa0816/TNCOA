# Efficient Exploration via Observation-Action Constraint on Trajectory-Based Intrinsic Reward

## Intro

This is an implementation of the method proposed in TNCOA



## Train on MiniGrid
```
nohup python src/train.py --game_name=ObstructedMaze-1Q --gpu 0  --features_dim 256 --model_features_dim 256 --n_steps 512 --num_processes 16 --n_epochs 3 --model_n_epochs 3 --batch_size 512 --ent_coef 0.0005 --adv_norm 0 --learning_rate 0.0005 --model_learning_rate 0.0005 --policy_cnn_norm 'LayerNorm' --policy_mlp_norm 'LayerNorm' --model_cnn_norm 'LayerNorm' --model_mlp_norm 'LayerNorm' --ext_rew_coef 100.0 --int_rew_coef 0.1 --rnd_err_norm 0 --run_id 0 --clip_range_vf 0.2 --group_name 'our_map'  > deir_1Q_our.log 2>&1 &

```

## Acknowledgements
Our vanilla RL algorithm is based on [DEIR](https://github.com/swan-utokyo/deir).