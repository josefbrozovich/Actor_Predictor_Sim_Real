import torch
import mamba_ssm

class RWR_Predict_cont:
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

        # critic configurations
        self.critic_model = hyperparams["critic_model"]
        self.critic_optimizer = hyperparams["critic_optimizer"]
        self.critic_features = hyperparams['critic_features']
        self.num_updates = hyperparams["num_updates"]
        self.eps = hyperparams["eps"]
        self.alpha = hyperparams["alpha"]

        self.critic_loss = torch.nn.MSELoss()

    def reset_state(self, batch_size):
        """
        Clear history
        """

        if self.model_type == "mamba":
            # making cache for stepping through
            self.inference_params = mamba_ssm.utils.generation.InferenceParams(max_seqlen=self.max_len, max_batch_size=batch_size)
            self.inference_params.seqlen_offset = 0
            self.inference_params.key_value_memory_dict = {}

            self.inference_params_predictor = mamba_ssm.utils.generation.InferenceParams(max_seqlen=self.max_len, max_batch_size=batch_size)
            self.inference_params_predictor.seqlen_offset = 0
            self.inference_params_predictor.key_value_memory_dict = {}

        elif self.model_type == "mamba_MLP_hybrid":
            
            self.inference_params_predictor = mamba_ssm.utils.generation.InferenceParams(max_seqlen=self.max_len, max_batch_size=batch_size)
            self.inference_params_predictor.seqlen_offset = 0
            self.inference_params_predictor.key_value_memory_dict = {}

        elif self.model_type == "transformer":
            self.cache_critic = None
            self.cache_actor = None
        
        elif self.model_type == "transformer_MLP_hybrid":
            self.cache_critic = None
            self.cache_actor = None
        

        # preallocating tensors
        # we are adding in features that the predictor tries to find out
        self.x_t = torch.zeros(batch_size, self.critic_features+self.state_space+self.action_space, device=self.device)
        self.trajectory = torch.zeros(batch_size, self.max_len, self.critic_features+self.state_space+self.action_space, device=self.device)
        # ations_taken has the same size
        self.action_taken = torch.zeros(batch_size, self.max_len, self.action_space, dtype=torch.long, device=self.device)
        self.step_idx = 0

    def get_action(self, state_vector, prev_actions):
        """
        Getting action during rollout, this should be done with no grad
        """
        with torch.no_grad():
            # current state
            # x_t = torch.cat([state_vector, prev_actions], dim=1).to(self.device)
            self.x_t[:, :self.state_space] = state_vector
            self.x_t[:, self.state_space:self.state_space+self.action_space] = prev_actions

            # checking if model is mamba
            # if self.model_type=="mamba":
                # # getting logits for predictor
                # _, current_h = self.critic_model.step(self.x_t[:, :self.state_space+self.action_space], self.inference_params_predictor)
                # self.inference_params_predictor.seqlen_offset += 1
                # # current_h has dimensions of (batch_size, state_space+action_space, states)
                # current_h = current_h.flatten(1)
                # # current_h has dimensions of (batch_size, (state_space+action_space)*states)
                # self.x_t[:, self.state_space+self.action_space:] = current_h
                # # current logits
                # logits, _ = self.model.step(self.x_t, self.inference_params)
                # self.inference_params.seqlen_offset += 1

            # elif self.model_type=="mamba_MLP_hybrid":
            if self.model_type=="mamba_MLP_hybrid":


                # getting logits for predictor
                _, current_h = self.critic_model.step(self.x_t[:, :self.state_space+self.action_space], self.inference_params_predictor)
                self.inference_params_predictor.seqlen_offset += 1
                # current_h has dimensions of (batch_size, state_space+action_space, states)
                current_h = current_h.flatten(1)
                self.x_t[:, self.state_space+self.action_space:] = current_h
                # current logits
                mu, sigma = self.model.forward(self.x_t)

            # model is transformer
            # elif self.model_type=="transformer":
            #     # getting logits for predictor
            #     _, current_h, self.cache_critic = self.critic_model.step(self.x_t[:,:self.state_space+self.action_space].unsqueeze(1), self.cache_critic)
            #     self.x_t[:, self.state_space+self.action_space:] = current_h.squeeze(1)
            #     # logits of trajectory
            #     logits, _, self.cache_actor = self.model.step(self.x_t.unsqueeze(1))
            #     # current logits
            #     logits = logits.squeeze(1)

            # elif self.model_type == "transformer_MLP_hybrid":
            #     # getting logits for predictor
            #     _, current_h, self.cache_critic = self.critic_model.step(self.x_t[:,:self.state_space+self.action_space].unsqueeze(1), self.cache_critic)
            #     self.x_t[:, self.state_space+self.action_space:] = current_h.squeeze(1)
            #     # logits of trajectory
            #     logits, _, = self.model.forward(self.x_t)
                
            else:

                raise ValueError("Model Type is not implemented")

            self.trajectory[:, self.step_idx, :] = self.x_t.detach()

            # get distribution for logits and temp
            # dist = torch.distributions.Categorical(logits=logits / self.temp)
            dist = torch.distributions.Normal(mu, sigma*self.temp)

            # sample action from distribution
            action = dist.sample()

            # append current action
            self.action_taken[:, self.step_idx, :] = action

            self.step_idx += 1

        return action
    
    def get_trajectory(self):
        # gets trajectory
        return self.trajectory[:, :self.step_idx, :]

    def update(self, rewards, terminates):
        
        trajectory = self.trajectory[:, :self.step_idx, :]
        action_taken = self.action_taken[:, :self.step_idx, :]
        rewards = rewards[:, :self.step_idx]
        terminates = terminates[:, :self.step_idx]

        B, N = rewards.shape
        # G is eposide length across the batch
        has_terminated = terminates.any(dim=1)
        dead = torch.where(has_terminated, terminates.float().argmax(dim=1), torch.full((B,), N, device=self.device))
        mask = torch.arange(N, device=self.device).expand(B, N) < dead.unsqueeze(1)

        # logits, _ = self.model.forward(trajectory)
        # we might have to change this
        mu, sigma, = self.model.forward(trajectory)

        dist = torch.distributions.Normal(mu, sigma*self.temp)

        # Group Advantage: Sum rewards only where mask is valid
        valid_rewards = (rewards * mask).sum(dim=1) 

        mu, sigma, = self.model.forward(trajectory)
        prob_traj = dist.log_prob(action_taken).sum(dim=-1)
        
        weights = torch.softmax(self.beta*valid_rewards, dim=0).detach()

        # loss
        loss = -(weights * prob_traj).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # getting loss of critics
        logits, _ = self.critic_model.forward(trajectory[:, :, :self.state_space+self.action_space])
        # logits have the shape (B, T, 1)

        # B
        satisfies = (rewards == 1).any(dim=1).float()

        # B, T, 1
        target_logits = satisfies[:, None, None].expand_as(logits).detach()

        loss_predictor = self.critic_loss(logits, target_logits)
        self.critic_optimizer.zero_grad()
        loss_predictor.backward()
        self.critic_optimizer.step()

        return loss_predictor.item()
    


    def update(self, rewards, terminates):
        """
        updating neural network
        """
        # termination at any point in the episode
        terminated_epidsode = torch.any(terminates, dim=1)
        # survival mask
        survival_mask = 1-terminated_epidsode.float()

        # actions to torch
        # truncating trajectory
        trajectory = self.trajectory[:, :self.step_idx, :]
        # trajectory is [B, T, features]
        action_taken = self.action_taken[:, :self.step_idx]
        rewards = rewards[:, :self.step_idx]

        # final reward
        final_reward = rewards.sum(dim=1)
        # adding survival mask

        final_reward = final_reward*survival_mask

        # getting loss of actor
        # logits
        mu, sigma = self.model.forward(trajectory)
        # distribution of actions
        dist = torch.distributions.Normal(mu, sigma*self.temp)
        # probabilitiy of choosing each action
        log_probs = dist.log_prob(action_taken)
        # weights of trajectory
        weights = torch.softmax(self.beta*final_reward, dim=0).detach()
        # probability of trajectory
        prob_traj = log_probs.sum(dim=(1, 2))
        loss = -(weights * prob_traj).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # getting loss of critics
        logits, _ = self.critic_model.forward(trajectory[:, :, :self.state_space+self.action_space])
        # logits have the shape (B, T, 1)

        # B
        satisfies = (rewards == 1).any(dim=1).float()

        # B, T, 1
        target_logits = satisfies[:, None, None].expand_as(logits).detach()

        loss_predictor = self.critic_loss(logits, target_logits)
        self.critic_optimizer.zero_grad()
        loss_predictor.backward()
        self.critic_optimizer.step()

        return loss_predictor.item()


