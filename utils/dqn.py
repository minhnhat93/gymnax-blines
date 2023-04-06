from functools import partial
import optax
import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
import gymnax
import dejax


class RolloutManager(object):
    def __init__(self, model, env_name, env_kwargs, env_params):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape
        self.apply_fn = model.apply
        self.select_action = self.select_action

    @partial(jax.jit, static_argnums=(0, 4))
    def select_action(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey,
        evaluation: bool = False,
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
        value = policy(train_state.apply_fn, train_state.params, obs, rng)
        action = jnp.max(value, axis=-1)
        # log_prob = pi.log_prob(action)
        return action, rng

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate(self, rng_input, train_state, num_envs):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, rng = self.select_action(train_state, obs, rng_net, evaluation=True)
            next_o, next_s, reward, done, _ = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action.squeeze(),
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = [
                next_o,
                next_s,
                train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ], [new_valid_mask]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                train_state,
                rng_episode,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )

        cum_return = carry_out[-2].squeeze()
        return jnp.mean(cum_return)


@partial(jax.jit, static_argnums=0)
def policy(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    state: jnp.ndarray,
    rng,
):
    value = apply_fn(params, state, rng)
    return value


def train_dqn(rng, config, model, params, mle_log):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    schedule_fn = optax.linear_schedule(
        init_value=-float(config.lr_begin),
        end_value=-float(config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    # TODO: add epsilon exploration
    epsilon_greedy = optax.linear_schedule(
        init_value = config.epsilon_start,
        end_value = config.epsilon_end,
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(
        model, config.env_name, config.env_kwargs, config.env_params
    )

    buffer = dejax.uniform_replay(config.buffer_size)
    state_space = rollout_manager.observation_space
    try:
        temp = state_space.shape[0]
        state_shape = state_space.shape
    except Exception:
        state_shape = [state_space]
    init_buffer_item = {
        "obs": jnp.empty(
            (config.num_envs, *state_shape), dtype=jnp.float32
        ),
        "action": jnp.empty(
            (config.num_envs,), dtype=jnp.float32
        ),
        "next_obs": jnp.empty(
            (config.num_envs, *state_shape), dtype=jnp.float32
        ),
        "reward": jnp.empty(
            (config.num_envs,), dtype=jnp.float32
        ),
        "done": jnp.empty(
            (config.num_envs,), dtype=jnp.float32
        ),
    }
    buffer_state = buffer.init_fn(init_buffer_item)

    target_model = params.copy({})

    @partial(jax.jit, static_argnums=4)
    def get_transition(
        train_state: TrainState,
        obs: jnp.ndarray,
        state: dict,
        rng: jax.random.PRNGKey,
        num_train_envs: int,
    ):
        action, new_key = rollout_manager.select_action(
            train_state, obs, rng, evaluation=False
        )
        # print(action.shape)
        new_key, key_step = jax.random.split(new_key)
        b_rng = jax.random.split(key_step, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(
            b_rng, state, action
        )
        return train_state, env_state, obs, action, next_obs, reward, done, new_key

    rng, rng_step, rng_reset, rng_eval, rng_update = jax.random.split(rng, 5)
    obs, env_state = rollout_manager.batch_reset(
        jax.random.split(rng_reset, config.num_train_envs)
    )

    total_steps = 0
    log_steps, log_return = [], []
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="DQN", leave=True)
    for step in t:
        train_state, env_state, obs, action, next_obs, reward, done, rng_step = get_transition(
            train_state,
            obs,
            env_state,
            rng_step,
            config.num_train_envs,
        )
        buffer_item = {"obs": obs, "action": action, "next_obs": next_obs, "reward": reward, "done": done}
        buffer_state = buffer.add_fn(buffer_state, buffer_item)

        total_steps += config.num_train_envs
        if isinstance(config.target_update, int):
            if (step + 1) % config.target_update == 0:
                target_model = train_state.params.copy({})
        else:
            target_model = target_model * (1 - config.target_update) + train_state.params * config.target_update

        if (step + 1) % config.update_freq == 0:
            batch = buffer.sample_fn(buffer_state, rng_update, config.batch_size)
            metric_dict, train_state, rng_update = update_epoch(
                train_state, target_model, config.gamma,
                batch["obs"], batch["action"], batch["next_obs"], batch["reward"], batch["done"]
            )

        if (step + 1) % config.evaluate_every_epochs == 0:
            rng, rng_eval = jax.random.split(rng)
            rewards = rollout_manager.batch_evaluate(
                rng_eval,
                train_state,
                config.num_test_rollouts,
            )
            log_steps.append(total_steps)
            log_return.append(rewards)
            t.set_description(f"R: {str(rewards)}")
            t.refresh()

            if mle_log is not None:
                mle_log.update(
                    {"num_steps": total_steps},
                    {"return": rewards},
                    model=train_state.params,
                    save=True,
                )

    return (
        log_steps,
        log_return,
        train_state.params,
    )


@jax.jit
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])

from tensorflow_probability.python.internal.backend.numpy.numpy_array import ops, _gather_nd_single, functools

def _gather_nd(  # pylint: disable=unused-argument
        params,
        indices,
        batch_dims=0,
        name=None):
    """gather_nd."""
    params = ops.convert_to_tensor(params)
    indices = ops.convert_to_tensor(indices, dtype_hint=np.int32)
    if batch_dims < 0:
        raise NotImplementedError('Negative `batch_dims` is currently unsupported.')
    gather_nd_ = _gather_nd_single
    gather_nd_ = functools.reduce(
        lambda g, f: f(g), [jax.vmap] * int(batch_dims),
        gather_nd_
    )
    return gather_nd_(params, indices)


def loss_actor_and_critic(
    params_model: flax.core.frozen_dict.FrozenDict,
    target_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    gamma: float,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    next_obs: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
) -> jnp.ndarray:

    value_pred = apply_fn(params_model, obs, rng=None)
    value_pred = _gather_nd(value_pred, action, batch_dims=-1)
    # value_pred = value_pred[:, 0]
    next_obs_value_pred = jax.lax.stop_gradient(jnp.max(apply_fn(target_model, next_obs, rng=None), axis=-1))
    target = reward + gamma * next_obs_value_pred * (1.0 - done)
    loss = 0.5 * jax.lax.pow(value_pred - target, 2).mean()

    # TODO: Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi

    return loss, (
        value_pred.mean(),
        target.mean(),
    )


@jax.jit
def update_epoch(
    train_state: TrainState,
    target_model: flax.core.frozen_dict.FrozenDict,
    gamma: float,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    next_obs: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
):
    # print(action[idx].shape, action[idx].reshape(-1, 1).shape)

    grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params,
        target_model,
        train_state.apply_fn,
        gamma,
        obs=obs, action=action, next_obs=next_obs, reward=reward, done=done,
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, total_loss
