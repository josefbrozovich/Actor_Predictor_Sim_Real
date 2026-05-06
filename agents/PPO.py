import torch
import mamba_ssm

class PPO:
    def __init__(self, model, optimizer, temp, max_len, hyperparams, model_type="mamba", device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.temp = temp
        self.max_len = max_len
        self.num_updates = hyperparams["num_updates"]
        self.eps = hyperparams["eps"]
        self.state_space = hyperparams["state_space"]
        self.action_space = hyperparams["action_space"]
        self.model_type = model_type
        self.device = torch.device(device)
        self.critic_model = hyperparams["critic_model"]
        self.critic_optimizer = hyperparams["critic_optimizer"]
        self.gamma = hyperparams["gamma"]
        self.alpha = hyperparams["alpha"]


    def reset_state(self, batch_size):
        """
        Clear history
        """
        if self.model_type == "mamba":
            # making cache for stepping through
            self.inference_params = mamba_ssm.utils.generation.InferenceParams(max_seqlen=self.max_len, max_batch_size=batch_size)
            self.inference_params.seqlen_offset = 0
            self.inference_params.key_value_memory_dict = {}

        # preallocating tensors
        self.x_t = torch.zeros(batch_size, self.state_space+self.action_space, device=self.device)
        self.trajectory = torch.zeros(batch_size, self.max_len, self.state_space+self.action_space, device=self.device)
        self.action_taken = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=self.device)
        self.step_idx = 0

    def get_action(self, state_vector, prev_actions):
        """
        Getting action during rollout, this should be done with no grad
        """
        # current state
        self.x_t[:, :self.state_space] = state_vector
        self.x_t[:, self.state_space:] = prev_actions

        self.trajectory[:, self.step_idx, :] = self.x_t

        # checking if model is mamba
        if self.model_type=="mamba":
            # current logits
            logits, _ = self.model.step(self.x_t, self.inference_params)
            self.inference_params.seqlen_offset += 1

        # model is transformer
        else:
            # current trajectory
            current_seq = self.trajectory[:, :self.step_idx+1, :]
            # logits of trajectory
            full_logits, _ = self.model.forward(current_seq)
            # current logits
            logits = full_logits[:, -1, :]

        # get distribution for logits and temp
        dist = torch.distributions.Categorical(logits=logits / self.temp)
        # sample action from distribution
        action = dist.sample()

        # append current action
        self.action_taken[:, self.step_idx] = action

        self.step_idx += 1

        return action

    def get_trajectory(self):
        # gets trajectory
        return self.trajectory[:, :self.step_idx, :]

    def update(self, rewards, terminates):
        trajectory = self.trajectory[:, :self.step_idx, :]
        action_taken = self.action_taken[:, :self.step_idx]
        rewards = rewards[:, :self.step_idx]
        terminates = terminates[:, :self.step_idx]

        with torch.no_grad():
            # critic values under old policy rollout
            values, _ = self.critic_model(trajectory)
            values = values.squeeze(-1)

            batch_size, seq_len = rewards.shape

            # simple discounted returns (no GAE)
            returns = torch.zeros_like(rewards)
            next_return = torch.zeros(batch_size, device=rewards.device)

            for t in reversed(range(seq_len)):
                next_non_terminal = 1.0 - terminates[:, t].float()
                returns[:, t] = rewards[:, t] + self.gamma * next_return * next_non_terminal
                next_return = returns[:, t]

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # old log probs from policy before updates
            logits, _ = self.model(trajectory)
            old_dist = torch.distributions.Categorical(logits=logits / self.temp)
            old_log_probs = old_dist.log_prob(action_taken)

        old_log_probs = old_log_probs.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        for _ in range(self.num_updates):
            # actor
            logits, _ = self.model(trajectory)
            dist = torch.distributions.Categorical(logits=logits / self.temp)
            new_log_probs = dist.log_prob(action_taken)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.alpha * entropy

            self.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            # critic
            current_values, _ = self.critic_model(trajectory)
            current_values = current_values.squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(current_values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
            self.critic_optimizer.step()

        return critic_loss.item()