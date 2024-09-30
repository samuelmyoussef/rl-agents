import numpy as np
import logging
from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

logger = logging.getLogger(__name__)


from sofagym.utils import copy_env, normalize_reward, load_env


class DeterministicNode(Node):
    def __init__(self, parent, planner, state=None, depth=0, node_num=0):
        super().__init__(parent, planner)
        self.state = state
        self.observation = None
        self.depth = depth
        self.reward = 0
        self.value_upper = 0
        self.value_lower = 0
        self.count = 1
        self.done = False
        self.node_num = node_num

    def selection_rule(self):
        if not self.children:
            return None
        actions = list(self.children.keys())
        index = self.random_argmax([self.children[a].get_value_lower_bound() for a in actions])
        return actions[index]

    def expand(self):
        self.planner.leaves.remove(self)
        if self.state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = self.state.get_available_actions()
        except AttributeError:
            actions = range(self.state.action_space.n)
        for action in actions:
            self.children[action] = type(self)(self,
                                               self.planner,
                                               #state=copy_env(self.state),
                                               state=load_env(self.state, self.node_num),
                                               depth=self.depth + 1,
                                               node_num=self.planner.nodes_num + 1)
            self.planner.nodes_num += 1
            observation, reward, done, info = self.planner.step(self.children[action].state, action)
            self.children[action].state.save_step(self.children[action].node_num)
            # reward = normalize_reward(reward)
            self.planner.leaves.append(self.children[action])
            self.children[action].update(reward, done, observation)

            # self.children[action].state.close()

    def update(self, reward, done, observation=None):
        if not np.all(0 <= reward) or not np.all(reward <= 1):
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")
        gamma = self.planner.config["gamma"]
        self.reward = reward
        self.observation = observation
        self.done = done
        self.value_lower = self.parent.value_lower + (gamma ** (self.depth - 1)) * reward
        self.value_upper = self.value_lower + (gamma ** self.depth) / (1 - gamma)
        if isinstance(done, np.ndarray):
            idx = np.where(done)
            next_value = self.value_lower[idx] + \
                         self.planner.config["terminal_reward"] * (gamma ** self.depth) / (1 - gamma)
            self.value_lower[idx] = next_value
            self.value_upper[idx] = next_value
        elif done:
            self.value_lower = self.value_upper = self.value_lower + \
                self.planner.config["terminal_reward"] * (gamma ** self.depth) / (1 - gamma)

        for node in self.sequence():
            node.count += 1

    def backup_values(self):
        if self.children:
            backup_children = [child.backup_values() for child in self.children.values()]
            self.value_lower = np.amax([b[0] for b in backup_children])
            self.value_upper = np.amax([b[1] for b in backup_children])
        return self.get_value_lower_bound(), self.get_value_upper_bound()

    def backup_to_root(self):
        if self.children:
            self.value_lower = np.amax([child.get_value_lower_bound() for child in self.children.values()])
            self.value_upper = np.amax([child.get_value_upper_bound() for child in self.children.values()])
            if self.parent:
                self.parent.backup_to_root()

    def get_value_lower_bound(self):
        return self.value_lower

    def get_value_upper_bound(self):
        return self.value_upper

    def get_value(self) -> float:
        return self.value_upper


class OptimisticDeterministicPlanner(AbstractPlanner):
    NODE_TYPE = DeterministicNode

    """
       An implementation of Optimistic Planning in Deterministic MDPs.
    """
    def __init__(self, env, config=None):
        super(OptimisticDeterministicPlanner, self).__init__(config)
        self.env = env
        self.leaves = None
        
        self.nodes_num = 0

    def reset(self):
        # if hasattr(self, 'leaves'):
        #     if self.leaves is not None:
        #         self.leaves.clear()
        
        self.nodes_num = 0
        self.root = self.NODE_TYPE(None, planner=self, node_num=self.nodes_num)
        self.leaves = [self.root]

    def run(self):
        """
            Run an OptimisticDeterministicPlanner episode
        """
        leaf_to_expand = max(self.leaves, key=lambda n: n.get_value_upper_bound())
        if leaf_to_expand.done:
            logger.warning("Expanding a terminal state")
        leaf_to_expand.expand()
        leaf_to_expand.backup_to_root()

    def plan(self, state, observation):
        self.root.state = state
        # self.root.state.reset()
        self.root.state.load_config(self.root.state.config_file)
        self.root.state = load_env(self.root.state, self.nodes_num)
        # self.root.state.save_step(self.nodes_num)
        for epoch in np.arange(self.config["budget"] // state.action_space.n):
            logger.debug("Expansion {}/{}".format(epoch + 1, self.config["budget"] // state.action_space.n))
            self.run()

        return self.get_plan()

    def step_by_subtree(self, action):
        super(OptimisticDeterministicPlanner, self).step_by_subtree(action)
        if not self.root.children:
            self.leaves = [self.root]
        #  v0 = r0 + g r1 + g^2 r2 +... and v1 = r1 + g r2 + ... = (v0-r0)/g
        for leaf in self.leaves:
            leaf.value_lower = (leaf.value_lower - self.root.reward) / self.config["gamma"]
            leaf.value_upper_bound = (leaf.value_upper_bound - self.root.reward) / self.config["gamma"]
        self.root.backup_values()


class DeterministicPlannerAgent(AbstractTreeSearchAgent):
    """
        An agent that performs optimistic planning in deterministic MDPs.
    """
    PLANNER_TYPE = OptimisticDeterministicPlanner
