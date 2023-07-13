from gym import Env
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper
from CybORG.Agents.Wrappers.GymnasiumWrapper import GymnasiumWrapper
import numpy as np
from CybORG.Simulator.Actions.Action import InvalidAction

class ChallengeWrapper(Env,BaseWrapper):
    def __init__(self, agent_name: str, env,
            reward_threshold=None, max_steps = None, use_action_masks=False, gymnasium=False):
        super().__init__(env)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        env = table_wrapper(env, output_mode='vector')
        if gymnasium:
            env = GymnasiumWrapper(agent_name=agent_name, env=env)
        else:
            env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None


        self.use_action_masks = use_action_masks
        self.action_masks = None
        self.episode_reward = 0
        # To do: automate compute for max rewards
        self.max_ep_reward = 14.7

    def step(self,action=None):
        obs, reward, done, info = self.env.step(action=action)
        if self.use_action_masks:
            self.action_masks = self._action_mask_fn()
        self.step_counter += 1
        self.episode_reward += reward
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True
        if self.episode_reward >= self.max_ep_reward:
            done = True
        last_action = info['action']
        if last_action.__str__().startswith('InvalidAction'):
            reward -= 0.1
        return obs, reward, done, info
    
    def _action_mask_fn(self):
        mask = np.full(self.action_space.n, True)

        for i, action in enumerate(self.env.possible_actions):
            inv_action = self.env.env.env.env.environment_controller.replace_action_if_invalid(action, self.env.env.env.env.environment_controller.agent_interfaces['Red'])
            if isinstance(inv_action, InvalidAction):
                mask[i] = False
        return mask

    def reset(self):
        self.step_counter = 0
        self.episode_reward = 0
        obs = self.env.reset()
        if self.use_action_masks:
            self.action_masks = self._action_mask_fn()
        return obs

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)

