import torch
import mamba_ssm

class unform:
    def __init__(self, model, optimizer, temp, max_len, hyperparams, model_type="mamba", device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.temp = temp
        self.max_len = max_len
        self.model_type = model_type
        self.device = torch.device(device)

        self.critic_loss = torch.nn.MSELoss()

    def reset_state(self, batch_size):
        """
        Clear history
        """

        # preallocating tensors
        # we are adding in features that the predictor tries to find out
        self.x_t = torch.zeros(batch_size, self.critic_features+self.state_space+self.action_space, device=self.device)
        self.trajectory = torch.zeros(batch_size, self.max_len, self.critic_features+self.state_space+self.action_space, device=self.device)
        # ations_taken has the same size
        self.action_taken = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=self.device)
        self.step_idx = 0


    def get_action(self, state_vector, prev_actions):
        """
        Getting action during rollout, this should be done with no grad
        """

        action = torch.randin(0, 4, (state_vector.shape[0]))

        self.step_idx += 1

        return action
    
    def get_trajectory(self):
        # gets trajectory
        return self.trajectory[:, :self.step_idx, :]

    def update(self, rewards, terminates):
        
        return None
    