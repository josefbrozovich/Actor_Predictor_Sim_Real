import torch

class cycler:
    def __init__(self, model_actor, model_critic, optimizer_actor, optimizer_critic, temp, max_len, hyperparams, model_type="cycler", device="cpu"):
        self.actor = model_actor
        self.optimizer_actor = optimizer_actor
        self.critic = model_critic
        self.optimizer_critic = optimizer_critic
        self.temp = temp
        self.max_len = max_len
        self.device = device
        self.num_updates = hyperparams["num_updates"]

    def reset_state(self, _):

        # history or state
        self.trajectory = []

        # history of actions
        self.action_taken = []

    def get_action(self, state_vector, prev_actions):
        """
        Getting action during rollout, this should be done with no grad
        """
        # current state
        x_t = torch.cat([state_vector]).to(self.device)

        self.trajectory.append(x_t.unsqueeze(1))

        # getting logits
        logits = self.model_actor(x_t)

        # get distribution for logits and temp
        dist = torch.distributions.Categorical(logits=logits/self.temp)
        # sample action from distribution
        action = dist.sample()

        # append current action
        self.action_taken.append(action)

        return action
    
    def update(self, reward, terminated):

        # added this
        # survival_mask = 1.0 - terminated.float()
        # rewards = list(torch.stack(rewards) * survival_mask)

        # go through rewards in cycler


        # actions to torch
        actions_seq = torch.stack(self.action_taken, dim=1).to(self.device).long()
        # trajectory to tensor
        trajectory = torch.cat(self.trajectory, dim=1).to(self.device)

        logits, _ = self.model.forward(trajectory)
        dist = torch.distributions.Categorical(logits=logits/self.temp)
        log_probs = dist.log_prob(actions_seq).sum(dim=1).detach()

        with torch.no_grad():
            old_log_probs = torch.stack(log_probs, dim=-1)
            actions_seq = torch.stack(self.action_taken, dim=-1)

        logits = self.actor.forward(trajectory)
        values = self.critic.forward(trajectory)

        values_shift = 0*values

        values_shift[:, :-1] = values[:, 1:]

        for _ in range(self.num_updates):
            logits = self.actor.forward(trajectory)
            values = self.critic.forward(trajectory)

            advantage = (reward_shape+self.gamma*values_shift-values).detach()

            # new probabilities of action
            dist = torch.distributions.Categorial(logits=logits/self.temp)
            # entropy of actor
            entropy_actor = dist.entropy()
            # log probabilities of choosing action
            new_log_probs = dist.log_prob(actions_seq).sum(dim=1)
            # getting ratio
            ratio = torch.exp(new_log_probs-old_log_probs)

            unclipped = ratio * advantage
            clipped   = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            ppo_loss = -(torch.min(unclipped, clipped)).mean()

            
            loss_actor = (ppo_loss-self.alpha*entropy_actor).mean()
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            value_target = (reward_shape+self.gamma*values_shift).detach()

            loss_critic = torch.nn.MSELoss(value_target, values)
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

