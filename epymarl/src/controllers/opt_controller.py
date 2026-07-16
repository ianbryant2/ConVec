from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class OptMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        opt_input_shape = self._get_opt_input_shape(scheme)
        self._build_agents(input_shape,opt_input_shape)

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.opt_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, opt_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        if self.args.action_selector in ["epsilon_greedy","noise_greedy"]:
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs],t_env, test_mode=test_mode)
        else:
            chosen_actions = self.action_selector.select_action(agent_outputs[bs],opt_outputs[bs], avail_actions[bs], t_env,test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        opt_inputs = self._build_opt_inputs(ep_batch, t)#
        agent_outs, self.hidden_states = self.agent(inputs, self.hidden_states)
        opt_outs, self.opt_hidden_states = self.opt(opt_inputs, self.opt_hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1),opt_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.opt_hidden_states = self.opt.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        param = list(self.agent.parameters())
        param += list(self.opt.parameters())
        return param

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.opt.load_state_dict(other_mac.opt.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.opt.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.opt.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.opt.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape,opt_input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.opt = agent_REGISTRY[self.args.agent](opt_input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_opt_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        if self.args.use_state:
            inputs.append(batch["state"][:, t].unsqueeze(-2).repeat(1,self.n_agents,1))
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_opt_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.use_state:
            input_shape += scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape