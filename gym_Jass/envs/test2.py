import argparse
import logging
import os
import random

import gym
#import torch.optim as optim

import optuna


#import pfrl
#from pfrl import experiments, explorers, q_functions, replay_buffers, utils
#from pfrl.agents.dqn import DQN
#from pfrl.experiments.evaluation_hooks import OptunaPrunerHook
from optuna.integration import TensorFlowPruningHook
#from pfrl.nn.mlp import MLP


def _objective_core(
    # optuna parameters
    trial,
    # training parameters
    env_id = "Jass-v0",
    outdir,
    monitor,
    gpu,
    steps,
    train_max_episode_len,
    eval_n_episodes,
    eval_interval,
    batch_size,
    # hyperparameters
    hyperparams,
)

    def make_env():
        env = suite_gym.make(env_id)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                "Supported only Box observation environments, but given: {}".format(
                    env.observation_space
                )
            )
        if len(env.observation_space.shape) != 1:
            raise ValueError(
                "Supported only observation spaces with ndim==1, but given: {}".format(
                    env.observation_space.shape
                )
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "Supported only discrete action environments, but given: {}".format(
                    env.action_space
                )
            )
        return env

    train_env = tf_py_environment.TFPyEnvironment(env_id)
    obs_space = train_env.observation_space
    obs_size = obs_space.low.size
    action_space = train_env.action_space
    n_actions = action_space.n

    q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params)


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    rbuf = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    partial_exponentially_decaying_eps = partial(exponentially_decaying_epsilon, train_step)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=partial_exponentially_decaying_eps,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step)

    eval_env = tf_py_environment.TFPyEnvironment(env_id)

    evaluation_hooks = [TensorFlowPruningHook(trial=trial)]
    _, eval_stats_history = experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=steps,
        eval_n_steps=None,
        eval_n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        eval_env=eval_env,
        train_max_episode_len=train_max_episode_len,
        evaluation_hooks=evaluation_hooks,
    )
    step = agent.train_step_counter.numpy()
    if step % eval_interval == 0:
        score = compute_avg_return(eval_env, agent.policy, num_eval_episodes)

    return score




def suggest(trial, steps):
    hyperparams = {}

    hyperparams["reward_scale_factor"] = trial.suggest_float(
        "reward_scale_factor", 1e-5, 10, log=True
    )
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)  # hyper-hyper-param
    hyperparams["hidden_sizes"] = []
    for n_channel in range(n_hidden_layers):
        # If n_channels is a large value, the precise number doesn't matter.
        # In other words, we should search over the smaller values more precisely.
        c = trial.suggest_int(
            "n_hidden_layers_{}_n_channels_{}".format(n_hidden_layers, n_channel),
            10,
            200,
            log=True,
        )
        hyperparams["hidden_sizes"].append(c)
    hyperparams["end_epsilon"] = trial.suggest_float("end_epsilon", 0.0, 0.3)
    max_decay_steps = steps // 2
    min_decay_steps = min(1e3, max_decay_steps)
    hyperparams["decay_steps"] = trial.suggest_int(
        "decay_steps", min_decay_steps, max_decay_steps
    )
    hyperparams["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # Adam's default eps==1e-8 but larger eps oftens helps.
    # (Rainbow: eps==1.5e-4, IQN: eps==1e-2/batch_size=3.125e-4)
    hyperparams["adam_eps"] = trial.suggest_float("adam_eps", 1e-8, 1e-3, log=True)
    inv_gamma = trial.suggest_float("inv_gamma", 1e-3, 1e-1, log=True)
    hyperparams["gamma"] = 1 - inv_gamma

    rbuf_capacity = steps
    min_replay_start_size = min(1e3, rbuf_capacity)
    # min: Replay start size cannot exceed replay buffer capacity.
    # max: decaying epsilon without training does not make much sense.
    max_replay_start_size = min(
        max(1e3, hyperparams["decay_steps"] // 2), rbuf_capacity
    )
    hyperparams["replay_start_size"] = trial.suggest_int(
        "replay_start_size",
        min_replay_start_size,
        max_replay_start_size,
    )
    # target_update_interval should be a multiple of update_interval
    hyperparams["update_interval"] = trial.suggest_int("update_interval", 1, 8)
    target_update_interval_coef = trial.suggest_int("target_update_interval_coef", 1, 4)
    hyperparams["target_update_interval"] = (
        hyperparams["update_interval"] * target_update_interval_coef
    )

    return hyperparams


