import torch
import mamba_ssm

class RWR:
    def __init__(self, model, optimizer, temp, max_len, hyperparams, model_type="mamba", device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.temp = temp
        self.max_len = max_len
        self.model_type = model_type
        self.device = torch.device(device)
        self.beta = hyperparams["beta"]
        self.state_space = hyperparams["state_space"]
        self.action_space = hyperparams["action_space"]

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
        with torch.no_grad():
            # current state
            # x_t = torch.cat([state_vector, prev_actions], dim=1).to(self.device)
            self.x_t[:, :self.state_space] = state_vector
            self.x_t[:, self.state_space:] = prev_actions

            self.trajectory[:, self.step_idx, :] = self.x_t.detach()

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
        """
        updating neural network
        """
        # termination at any point in the episode
        terminated_epidsode = torch.any(terminates, dim=1)
        # survival mask
        survival_mask = 1-terminated_epidsode.float()

        # actions to torch
        # truncating trajectiry
        trajectory = self.trajectory[:, :self.step_idx, :]
        action_taken = self.action_taken[:, :self.step_idx]


        # final reward
        final_reward = rewards.sum(dim=1)
        # adding survival mask

        final_reward = final_reward*survival_mask

        # logits
        logits, _ = self.model.forward(trajectory)
        # distribution of actions
        dist = torch.distributions.Categorical(logits=logits/self.temp)
        # probabilitiy of choosing each action
        log_probs = dist.log_prob(action_taken)
        # weights of trajectory
        weights = torch.softmax(self.beta*final_reward, dim=0).detach()
        # probability of trajectory
        prob_traj = log_probs.sum(dim=1)
        # loss
        loss = -(weights * prob_traj).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
