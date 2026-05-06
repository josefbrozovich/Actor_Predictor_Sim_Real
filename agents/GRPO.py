import torch
import mamba_ssm

class GRPO:
    def __init__(self, model, optimizer, temp, max_len, hyperparams, model_type="mamba", device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.temp = temp
        self.max_len = max_len
        self.num_updates = hyperparams["num_updates"]
        self.beta = hyperparams["beta"]
        self.eps = hyperparams["eps"]
        self.state_space = hyperparams["state_space"]
        self.action_space = hyperparams["action_space"]
        self.model_type = model_type
        self.device = torch.device(device)

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
        # ... (keep your trajectory/action slicing logic) ...
        
        trajectory = self.trajectory[:, :self.step_idx, :]
        action_taken = self.action_taken[:, :self.step_idx]
        rewards = rewards[:, :self.step_idx]
        terminates = terminates[:, :self.step_idx]
        B, N = rewards.shape

        # G is eposide length across the batch
        has_terminated = terminates.any(dim=1)
        G = torch.where(has_terminated, terminates.float().argmax(dim=1), torch.full((B,), N, device=self.device))
        mask = torch.arange(N, device=self.device).expand(B, N) < G.unsqueeze(1)

        with torch.no_grad():
            logits, _ = self.model.forward(trajectory)
            dist = torch.distributions.Categorical(logits=logits/self.temp)
            old_log_probs = dist.log_prob(action_taken)

            # Group Advantage: Sum rewards only where mask is valid
            valid_rewards = (rewards * mask).sum(dim=1) 
            mean_r = valid_rewards.mean()
            std_r = valid_rewards.std() + 1e-8
            # Advantage becomes a B x 1 tensor that broadcasts across the sequence
            advantage = ((valid_rewards - mean_r) / std_r).unsqueeze(1)

        for _ in range(self.num_updates):
            logits, _ = self.model.forward(trajectory)
            dist = torch.distributions.Categorical(logits=logits/self.temp)
            new_log_probs = dist.log_prob(action_taken)
            
            # Ratio for PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # KL Divergence
            # Using the log-space version for better stability
            # log_ratio = old_log_probs - new_log_probs
            # kl = torch.exp(log_ratio) - log_ratio - 1

            # Combine: minimize negative (surrogate - kl_penalty)
            per_token_loss = -torch.min(surr1, surr2)
            
            # Apply mask so we don't learn from steps after the robot "died"
            loss = (per_token_loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return 1.0