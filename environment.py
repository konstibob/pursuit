import numpy as np
from pettingzoo.sisl import pursuit_v4

class PursuitEnvWrapper:
    def __init__(self, config):
        valid_keys = ["max_cycles", "x_size", "y_size", "shared_reward", "n_evaders", "n_pursuers",
                      "obs_range", "n_catch", "freeze_evaders", "tag_reward", "catch_reward",
                      "urgency_reward", "surround", "constraint_window", "render_mode"]

        env_kwargs = {k: v for k, v in config.items() if k in valid_keys}

        self.env = pursuit_v4.parallel_env(**env_kwargs)
        self._game = self.env.aec_env.env.env.env
        self.env.reset()

        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        self.n_pursuers = config["n_pursuers"]
        self.n_evaders = config["n_evaders"]

        self.n_actions = self.env.action_space(self.agents[0]).n

        sample_obs = self.env.observation_space(self.agents[0]).shape
        self.obs_dim = np.prod(sample_obs)

        self.state_dim = 2 * self.n_pursuers + 2 * self.n_evaders

    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._process_obs(obs_dict)

    def step(self, actions):
        act_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}

        obs_dict, rewards_dict, terms, truncs, infos = self.env.step(act_dict)

        obs, state = self._process_obs(obs_dict)
        reward = sum(rewards_dict.values()) / self.n_agents
        done = any(terms.values()) or any(truncs.values())

        return obs, state, reward, done, infos

    def _process_obs(self, obs_dict):
        obs_list = []
        for agent in self.agents:
            flat_o = obs_dict[agent].flatten()
            obs_list.append(flat_o)

        obs = np.array(obs_list, dtype=np.float32)

        game = self._game

        pursuer_state = game.pursuer_layer.get_state().astype(np.float32)
        pursuer_state[0::2] /= (game.x_size - 1)
        pursuer_state[1::2] /= (game.y_size - 1)

        evader_state = np.zeros(2 * game.n_evaders, dtype=np.float32)
        live_idx = 0
        for orig_idx in range(game.n_evaders):
            if not game.evaders_gone[orig_idx]:
                x, y = game.evader_layer.get_position(live_idx)
                evader_state[2 * orig_idx]     = x / (game.x_size - 1)
                evader_state[2 * orig_idx + 1] = y / (game.y_size - 1)
                live_idx += 1

        state = np.concatenate([pursuer_state, evader_state])

        return obs, state

    def close(self):
        self.env.close()
