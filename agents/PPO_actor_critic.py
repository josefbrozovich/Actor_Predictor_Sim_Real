import torch
# import mamba_ssm

class PPO_actor_critic:
    def __init__(self, model, optimizer, temp, max_len, hyperparams, model_type="mamba", device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.temp = temp
        self.max_len = max_len
        self.num_updates = hyperparams["num_updates"]
        self.eps = hyperparams["eps"]
        self.gamma = hyperparams["gamma"]
        self.alpha = hyperparams["alpha"]

        self.state_space = hyperparams["state_space"]
        self.action_space = hyperparams["action_space"]

        self.critic_model = hyperparams["critic_model"]
        self.critic_optimizer = hyperparams["critic_optimizer"]

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
        
        if self.model_type=="MLP":
            self.trajectory = torch.zeros(batch_size, self.max_len, self.state_space, device=self.device)
        else:
            raise Exception("MLP is onyl implementation for PPO_actor_critc")

        self.action_taken = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=self.device)
        self.step_idx = 0

    def get_action(self, state_vector, prev_actions):
        """
        Getting action during rollout, this should be done with no grad
        """
        # current state
        # self.x_t[:, :self.state_space] = state_vector
        # self.x_t[:, self.state_space:] = prev_actions

        self.trajectory[:, self.step_idx, :] = state_vector

        # checking if model is mamba
        if self.model_type=="mamba":
            # current logits
            logits, _ = self.model.step(self.x_t, self.inference_params)
            self.inference_params.seqlen_offset += 1

        # model is transformer
        elif self.model_type=="MLP":
            # current trajectory
            current_seq = self.trajectory[:, self.step_idx, :]
            # logits of trajectory
            logits, _ = self.model.forward(current_seq)

        # model is transformer
        elif self.model_type=="transformer":
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
        r_t = rewards[:, :self.step_idx]
        actions = self.action_taken[:, :self.step_idx]
        term_mask = terminates[:, :self.step_idx] 
        alive_mask = (term_mask == 0).float()

        with torch.no_grad():
            # Old Policy Log Probs
            logits_old, _ = self.model.forward(trajectory)
            dist_old = torch.distributions.Categorical(logits=logits_old / self.temp)
            old_log_probs = dist_old.log_prob(actions).detach()

            # Old Value Estimates
            values_old, _ = self.critic_model.forward(trajectory)
            values_old = values_old.squeeze(-1).detach()

            # GAE (Generalized Advantage Estimation)
            # This is critical for sparse rewards so the final reward propagates backwards!
            advantages = torch.zeros_like(r_t)
            lastgaelam = 0
            
            # GAE Hyperparameter (usually 0.95)
            lam = 0.95 

            # Calculate backwards
            for t in reversed(range(self.step_idx)):
                if t == self.step_idx - 1:
                    nextnonterminal = 1.0 - term_mask[:, t].float()
                    nextvalues = 0.0 # Assume 0 if it's the absolute end of the rollout
                else:
                    nextnonterminal = 1.0 - term_mask[:, t].float()
                    nextvalues = values_old[:, t + 1]
                
                delta = r_t[:, t] + self.gamma * nextvalues * nextnonterminal - values_old[:, t]
                advantages[:, t] = lastgaelam = delta + self.gamma * lam * nextnonterminal * lastgaelam

            # Returns are just Advantages + Values
            returns = advantages + values_old

            # Normalize advantages (Standard PPO stabilization)
            adv_mean = (advantages * alive_mask).sum() / (alive_mask.sum() + 1e-8)
            adv_std = torch.sqrt(((advantages - adv_mean)**2 * alive_mask).sum() / (alive_mask.sum() + 1e-8))
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # ---------------------------------------------------------
        # 2. PPO OPTIMIZATION EPOCHS
        # ---------------------------------------------------------
        for _ in range(self.num_updates):
            # Recalculate current logits and values
            logits, _ = self.model.forward(trajectory)
            values, _ = self.critic_model.forward(trajectory)
            values = values.squeeze(-1)

            dist = torch.distributions.Categorical(logits=logits / self.temp)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Actor Loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            
            ppo_loss = -torch.min(surr1, surr2)
            loss_ppo = (ppo_loss * alive_mask).sum() / (alive_mask.sum() + 1e-8)
            loss_actor = loss_ppo - self.alpha * (entropy * alive_mask).sum() / (alive_mask.sum() + 1e-8)

            self.optimizer.zero_grad(set_to_none=True)
            loss_actor.backward()
            self.optimizer.step()

            # Critic Loss (MSE against fixed Returns)
            loss_critic = (((returns - values) ** 2) * alive_mask).sum() / (alive_mask.sum() + 1e-8)
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss_critic.backward()
            self.critic_optimizer.step()

            return 1.0