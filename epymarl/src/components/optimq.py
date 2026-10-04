"""OptimQ: learned joint-action critics Q*_i, giving the concession delta.

Q*_i(s, a) is agent i's optimal value of its own reward when the joint action
is a single action; delta_i = max_a Q*_i(s, a) - Q*_i(s, a_t).

- Two critics, each trained on its own half of the data, combined with the
  double estimator so the max is not biased upwards.
- delta is not clamped at 0, so estimation noise cancels over an episode.
- The step index t / time_limit is an input, making Q* finite-horizon.
- Off-policy double-Q TD targets with soft-updated target nets.
- Optional stabilisers: Huber loss, LayerNorm, and reward_range, which clamps
  bootstrapped values to the range Q* can take.
- Optional prioritised replay (replay.prioritized): each critic samples its
  buffer by TD error (torchrl's segment-tree buffer), with importance weights.

Output size grows as n_actions ** n_agents.
"""
import copy
import itertools

import torch as th
import torch.nn as nn
import torch.nn.functional as F

MAX_JOINT_ACTIONS = 100_000


class OptimQ:
    def __init__(
        self,
        args,
        state_dim,
        hidden_dim,
        lr,
        buffer_size,
        batch_size,
        target_tau,
        grad_clip,
        loss,
        layer_norm,
        reward_range,
        replay=None,
    ):
        # Defaults live in config/envs/gymma_budget.yaml (budget.concession_args.optimq);
        # its data and training blocks go to components/concession_training.py.
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.device = args.device
        # Budget runs are gymma envs, whose episodes end at time_limit.
        self.horizon = args.env_args["time_limit"]

        losses = {"mse": F.mse_loss, "huber": F.smooth_l1_loss}
        if loss not in losses:
            raise ValueError(f"Unknown OptimQ loss '{loss}' (available: {sorted(losses)})")
        self.loss_fn = losses[loss]

        # value_bounds[k][r]: lower (k=0) / upper (k=1) bound of Q* with r steps left.
        self.value_bounds = None
        if reward_range is not None:
            r_min, r_max = reward_range
            discounts = self.gamma ** th.arange(self.horizon, device=self.device).float()
            horizon_sums = th.cat([discounts.new_zeros(1), discounts.cumsum(0)])
            self.value_bounds = (r_min * horizon_sums, r_max * horizon_sums)

        n_joint = self.n_actions**self.n_agents
        if n_joint > MAX_JOINT_ACTIONS:
            raise ValueError(
                f"OptimQ enumerates joint actions: {self.n_actions}^{self.n_agents} "
                f"= {n_joint} is more than {MAX_JOINT_ACTIONS}"
            )
        # joint_actions[j] is the per-agent action tuple of joint action j.
        self.joint_actions = th.tensor(
            list(itertools.product(range(self.n_actions), repeat=self.n_agents)),
            device=self.device,
        )
        self.radix = self.n_actions ** th.arange(
            self.n_agents - 1, -1, -1, device=self.device
        )

        self.critics = [
            self._make_critic(state_dim + 1, hidden_dim, n_joint, lr, layer_norm)
            for _ in range(2)
        ]
        # Each transition trains only one critic, so their estimation errors are
        # independent; with shared data both would fit the same noisy averages and
        # the double estimator would reduce to the single one.
        replay = replay or {}
        if replay.get("prioritized"):
            self.buffers = [
                _PrioritizedTransitionBuffer(buffer_size // 2, batch_size, replay["alpha"],
                                             replay["beta"], self.device)
                for _ in self.critics
            ]
        else:
            self.buffers = [
                _TransitionBuffer(buffer_size // 2, self.device) for _ in self.critics
            ]
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.grad_clip = grad_clip

    def _make_critic(self, input_dim, hidden_dim, n_joint, lr, layer_norm):
        """One critic; subclasses swap the network (see OptimQAdditiveLinear)."""
        return _Critic(input_dim, hidden_dim, self.n_agents, n_joint, lr, layer_norm, self.device)

    def _inputs(self, states, t):
        """Critic input: the state with the step index t (B,) appended as t / horizon."""
        time = t.float().view(-1, 1) / self.horizon
        return th.cat([states.float(), time], dim=-1)

    def _joint_index(self, actions):
        """(B, n_agents) per-agent actions -> (B, n_agents, 1) joint index for gather."""
        joint = (actions.long() * self.radix).sum(-1)
        return joint.view(-1, 1, 1).expand(-1, self.n_agents, 1)

    def _joint_avail(self, avail_actions):
        """(B, n_agents, n_actions) per-agent mask -> (B, 1, n_joint) joint mask."""
        agents = th.arange(self.n_agents, device=self.device)
        per_agent = avail_actions[:, agents, self.joint_actions]  # (B, n_joint, n_agents)
        return per_agent.bool().all(-1).unsqueeze(1)

    @th.no_grad()
    def __call__(self, states, actions, avail_actions, t):
        """Concession delta (B, n_agents) for joint actions taken at states at step t."""
        inputs = self._inputs(states, th.full((len(states),), t, device=states.device))
        avail = self._joint_avail(avail_actions)
        taken = self._joint_index(actions)
        q1, q2 = (c.q(c.target_net, inputs) for c in self.critics)
        best1 = q1.masked_fill(~avail, -1e9).argmax(-1, keepdim=True)
        best2 = q2.masked_fill(~avail, -1e9).argmax(-1, keepdim=True)
        delta = 0.5 * (q2.gather(-1, best1) - q2.gather(-1, taken)) + 0.5 * (
            q1.gather(-1, best2) - q1.gather(-1, taken)
        )
        return delta.squeeze(-1)

    def add(self, transitions):
        """Split new transitions at random between the two critics' buffers."""
        to_first = th.rand(len(transitions["state"]), device=self.device) < 0.5
        for buffer, rows in zip(self.buffers, (to_first, ~to_first)):
            if rows.any():
                buffer.add({k: v[rows] for k, v in transitions.items()})

    def train(self, n_steps, transitions):
        """Take n_steps gradient steps per critic. Returns stats to log, or
        None while a buffer is smaller than batch_size."""
        if n_steps < 1 or min(len(b) for b in self.buffers) < self.batch_size:
            return None
        steps = [
            self._train_step(critic, buffer)
            for _ in range(n_steps)
            for critic, buffer in zip(self.critics, self.buffers)
        ]
        losses, clipped = zip(*steps)
        stats = {"concession_loss": sum(losses) / len(losses)}
        if transitions is not None and len(transitions["state"]):
            with th.no_grad():
                inputs = self._inputs(transitions["state"], transitions["t"])
                taken = self._joint_index(transitions["actions"])
                q1, q2 = (c.q_at(c.target_net, inputs, taken) for c in self.critics)
            # Large gaps mean noisy Q* estimates, and so noisy deltas.
            stats["concession_critic_gap"] = (q1 - q2).abs().mean().item()
        if self.value_bounds is not None:
            # Share of bootstrapped next values outside the reward_range bounds.
            stats["concession_clip_frac"] = sum(clipped) / len(clipped)
        return stats

    def _train_step(self, critic, buffer):
        b = buffer.sample(self.batch_size)
        inputs = self._inputs(b["state"], b["t"])
        next_inputs = self._inputs(b["next_state"], b["t"] + 1)
        q_taken = critic.q_at(critic.net, inputs, self._joint_index(b["actions"])).squeeze(-1)

        with th.no_grad():
            next_avail = self._joint_avail(b["next_avail"])
            # Terminal rows are masked out by (1 - done); keep their max finite.
            next_avail = next_avail | ~next_avail.any(-1, keepdim=True)
            next_q = critic.q(critic.net, next_inputs).masked_fill(~next_avail, -1e9)
            best_next = next_q.argmax(-1, keepdim=True)  # online net selects
            next_value = critic.q_at(critic.target_net, next_inputs, best_next)
            next_value = next_value.squeeze(-1)
            clipped = 0.0
            if self.value_bounds is not None:
                remaining = (self.horizon - 1 - b["t"]).long().clamp(min=0)
                low, high = (bound[remaining].unsqueeze(-1) for bound in self.value_bounds)
                clipped = ((next_value < low) | (next_value > high)).float().mean().item()
                next_value = th.minimum(th.maximum(next_value, low), high)
            target = b["env_reward"] + self.gamma * (1 - b["terminated"]) * next_value

        # Mean over agents and samples; prioritised samples carry importance
        # weights (1 for the uniform buffer, which makes this the plain mean).
        per_sample = self.loss_fn(q_taken, target, reduction="none").mean(-1)
        loss = (b["weight"] * per_sample).mean() if "weight" in b else per_sample.mean()
        critic.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.net.parameters(), self.grad_clip)
        critic.optimiser.step()
        if "index" in b:
            # A transition's priority is its worst-fit agent head.
            buffer.update(b["index"], (q_taken - target).detach().abs().amax(-1))

        with th.no_grad():
            for p, tp in zip(critic.net.parameters(), critic.target_net.parameters()):
                tp.mul_(1 - self.target_tau).add_(self.target_tau * p)
        return loss.item(), clipped


class _Critic:
    """One joint-action critic: online net, soft-updated target net, optimiser."""

    def __init__(self, state_dim, hidden_dim, n_agents, n_joint, lr, layer_norm, device):
        self.n_agents = n_agents

        def hidden(in_dim):
            norm = [nn.LayerNorm(hidden_dim)] if layer_norm else []
            return [nn.Linear(in_dim, hidden_dim), *norm, nn.ReLU()]

        self.net = nn.Sequential(
            *hidden(state_dim),
            *hidden(hidden_dim),
            nn.Linear(hidden_dim, n_agents * n_joint),
        ).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimiser = th.optim.Adam(self.net.parameters(), lr=lr)

    def q(self, net, inputs):
        """(B, input_dim) -> (B, n_agents, n_joint)"""
        return net(inputs).view(inputs.shape[0], self.n_agents, -1)

    def q_at(self, net, inputs, joint):
        """Values at joint (B, n_agents, 1) -> (B, n_agents, 1)."""
        return self.q(net, inputs).gather(-1, joint)


class _TransitionBuffer:
    """Fixed-size FIFO of transitions, stored as tensors on one device."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.data = None
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, transitions):
        n = len(transitions["state"])
        if self.data is None:
            self.data = {
                k: th.zeros((self.capacity,) + v.shape[1:], dtype=v.dtype, device=self.device)
                for k, v in transitions.items()
            }
        idx = (self.pos + th.arange(n, device=self.device)) % self.capacity
        for k, v in transitions.items():
            self.data[k][idx] = v.to(self.device)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = th.randint(self.size, (batch_size,), device=self.device)
        return {k: v[idx] for k, v in self.data.items()}


class _PrioritizedTransitionBuffer:
    """Fixed-size FIFO sampled by priority (torchrl's segment-tree buffer).

    New transitions enter at the highest priority seen so far, so each is
    sampled soon after it arrives; training sets priorities to TD errors.
    alpha skews sampling towards high priorities (0 = uniform); beta sets how
    much the importance weights undo that skew in the loss (0 = not at all).
    """

    def __init__(self, capacity, batch_size, alpha, beta, device):
        from tensordict import TensorDict
        from torchrl.data.replay_buffers import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

        self._TensorDict = TensorDict
        self.memory = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            eps=1e-6,
            priority_key="td_error",
            storage=LazyTensorStorage(max_size=capacity, device=device),
            batch_size=batch_size,
        )
        self.device = device
        self.batch_size = batch_size
        self.max_priority = th.tensor(1.0, device=device)

    def __len__(self):
        return len(self.memory)

    def add(self, transitions):
        n = len(transitions["state"])
        data = {k: v.to(self.device) for k, v in transitions.items()}
        data["td_error"] = self.max_priority.expand(n).clone()
        self.memory.extend(self._TensorDict(data, batch_size=[n]))

    def sample(self, batch_size):
        if batch_size != self.batch_size:
            raise ValueError(f"buffer built for batches of {self.batch_size}, asked for {batch_size}")
        batch = self.memory.sample()
        out = {k: v for k, v in batch.items() if k not in ("td_error", "_weight")}
        out["weight"] = batch["_weight"].to(self.device).float()
        return out

    def update(self, index, priority):
        self.max_priority = th.maximum(self.max_priority, priority.max())
        self.memory.update_priority(index, priority)
