import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

def weight_init(m):
    """  
    Initializes the weights of a neural network layer using the Xavier normal distribution and sets biases to zero.  
  
    Parameters  
    ----------  
    m : torch.nn.Module  
        The neural network layer (usually `nn.Linear`) to initialize.  
  
    Notes  
    -----  
    This function checks if the input module `m` is an instance of `nn.Linear`. If so, it initializes  
    the weights (`m.weight`) using the Xavier normal distribution (also known as Glorot normal initialization)  
    and sets the biases (`m.bias`) to zero. This initialization technique is designed to keep  
    the weights of the neural network layers within a reasonable range during training, helping  
    with convergence and preventing vanishing or exploding gradients.  
    """  
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class Policynet(nn.Module):
    """  
    Policy network for generating actions and corresponding log-probabilities.  
  
    Attributes  
    ----------  
    nn : torch.nn.Sequential  
        A neural network that processes the input state and produces a hidden representation.  
    fc_mu : torch.nn.Linear  
        A fully connected layer that maps the hidden representation to the mean of the action distribution.  
    fc_logstd : torch.nn.Linear  
        A fully connected layer that maps the hidden representation to the log standard deviation of the action distribution.  
  
    Methods  
    -------  
    forward(x)  
        Generates actions and corresponding log-probabilities given an input state.  
  
    Parameters  
    ----------  
    state_dim : int  
        Dimensionality of the input state.  
    hidden_dim : int  
        Dimensionality of the hidden representation.  
    action_dim : int  
        Dimensionality of the action space.  
    action_space : tuple
        Tuple indicating the minimum and maximum action values (min_action, max_action).
    Notes  
    -----  
    The `forward` method generates actions by sampling from a multivariate normal distribution  
    parameterized by the mean (`mu`) and standard deviation (`std`). The covariance matrix is  
    constructed as a diagonal matrix with the elements of `std` on the diagonal. Actions are  
    sampled using the `rsample` method to allow for gradient propagation through the sampling process.  
  
    The log-probability of the generated actions is also computed and returned.  
    """  
    def __init__(self, state_dim ,hidden_dim, action_dim, action_space):
        super(Policynet, self).__init__()
        self.nn = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU()
                               )
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        self.min_action, self.max_action = action_space
        self.apply(weight_init)
        
    def forward(self, x):
        """  
        Generates actions and corresponding log-probabilities given an input state.  
  
        Parameters  
        ----------  
        x : torch.Tensor  
            Input state tensor of shape `(batch_size, state_dim)`.  
  
        Returns  
        -------  
        action : torch.Tensor  
            Generated actions of shape `(batch_size, action_dim)`.  
        logprob : torch.Tensor  
            Log-probabilities of the generated actions of shape `(batch_size, 1)`.  
        """  
        x = self.nn(x)
        mu = torch.tanh(self.fc_mu(x))
        mu = self.min_action + 0.5 * (mu + 1.0) * (self.max_action - self.min_action)
        logstd = self.fc_logstd(x)
        std = F.softplus(logstd) + 1e-6
        cov = torch.stack([torch.diag(s) for s in std])
        mn = MultivariateNormal(mu, cov)
        action = mn.rsample()
        logprob = mn.log_prob(action)
        return action, logprob.view(-1,1)

class Qnet(nn.Module):
    """  
    Q-network for estimating the state-action value in reinforcement learning.  
  
    Attributes  
    ----------  
    nn : torch.nn.Sequential  
        A neural network that processes the concatenated state and action and outputs the Q-value.  
  
    Methods  
    -------  
    forward(x, a)  
        Computes the Q-value given a state and an action.  
  
    Parameters  
    ----------  
    state_dim : int  
        Dimensionality of the state space.  
    hidden_dim : int  
        Dimensionality of the hidden layers in the neural network.  
    action_dim : int  
        Dimensionality of the action space.  
  
    Notes  
    -----  
    The `forward` method concatenates the state `x` and action `a` along the second dimension,  
    then passes the concatenated vector through the neural network `nn` to obtain the Q-value.  
    """  
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.nn = nn.Sequential(nn.Linear(state_dim+action_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, 1)
                               )
        self.apply(weight_init)
        
    def forward(self, x, a):
        """  
        Computes the Q-value given a state and an action.  
  
        Parameters  
        ----------  
        x : torch.Tensor  
            Input state tensor of shape `(batch_size, state_dim)`.  
        a : torch.Tensor  
            Input action tensor of shape `(batch_size, action_dim)`.  
  
        Returns  
        -------  
        q_value : torch.Tensor  
            The computed Q-values of shape `(batch_size, 1)`.  
        """  
        cat = torch.cat([x, a], dim=1)
        return self.nn(cat)
        
class SAC:
    """  
    Implementation of the Soft Actor-Critic (SAC) algorithm for reinforcement learning.  

    Attributes  
    ----------  
    actor : Policynet  
        The policy network that outputs actions given states.  
    critic_1, critic_2 : Qnet  
        Two Q-networks (also known as critics) that estimate the state-action value.  
    target_critic_1, target_critic_2 : Qnet  
        Target Q-networks used for stabilizing learning via soft updates.  
    actor_optimizer : torch.optim.Optimizer  
        Optimizer for updating the actor network.  
    critic_1_optimizer, critic_2_optimizer : torch.optim.Optimizer  
        Optimizers for updating the two critic networks.  
    log_alpha : torch.Tensor  
        Learnable temperature parameter for entropy regularization.  
    log_alpha_optimizer : torch.optim.Optimizer  
        Optimizer for updating the temperature parameter.  
    target_entropy : float  
        Target entropy used for entropy regularization.  
    gamma : float  
        Discount factor for future rewards.  
    tau : float  
        Soft update coefficient for target networks.  
    device : torch.device  
        Device (CPU or GPU) on which the networks and tensors should be stored.  

    Methods  
    -------  
    take_action(state)  
        Given a state, returns an action sampled from the actor network.  
    calc_target(rewards, next_states, dones)  
        Computes the target Q-values for a batch of transitions.  
    soft_update(net, target_net)  
        Updates the target network towards the main network using a soft update rule.  
    update(transition_dict)  
        Performs a training update using a batch of transitions.  
    """  
    def __init__(self, state_dim, hidden_dim, action_dim, action_space, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        """  
        Initialize the SAC agent.  
  
        Parameters  
        ----------  
        state_dim : int  
            Dimensionality of the state space.  
        hidden_dim : int  
            Dimensionality of the hidden layers in the neural networks.  
        action_dim : int  
            Dimensionality of the action space.  
        actor_lr : float  
            Learning rate for the actor.  
        critic_lr : float  
            Learning rate for the critics.  
        alpha_lr : float  
            Learning rate for the temperature parameter alpha.  
        target_entropy : float  
            Target entropy for the policy.  
        tau : float  
            Soft update factor for the target networks.  
        gamma : float  
            Discount factor.  
        device : str or torch.device  
            Device on which to run the computations (e.g., 'cuda' or 'cpu').  
        """  
        self.actor                   = Policynet(state_dim, hidden_dim, action_dim, action_space).to(device) 
        self.critic_1                = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2                = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1         = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2         = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer         = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer      = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer      = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha               = torch.tensor(np.log(.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer     = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy          = target_entropy
        self.gamma                   = gamma
        self.tau                     = tau
        self.device                  = device
        
    def take_action(self, state):
        """  
        Take an action given the current state.  
  
        Parameters  
        ----------  
        state : array_like  
            Current state of the environment.  
  
        Returns  
        -------  
        action : array_like  
            Action taken by the agent.  
        """  
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.actor(state)[0].detach().cpu().numpy()
    
    def calc_target(self, rewards, next_states, dones):
        """  
        Calculate the TD targets for the critics.  
  
        Parameters  
        ----------  
        rewards : array_like  
            Rewards received from the environment.  
        next_states : array_like  
            Next states observed from the environment.  
        dones : array_like  
            Boolean array indicating whether each episode has terminated.  
  
        Returns  
        -------  
        td_target : torch.Tensor  
            Temporal difference targets for the critics.  
        """  
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def soft_update(self, net, target_net):
        """  
        Perform a soft update of the target network parameters.  
  
        Parameters  
        ----------  
        net : nn.Module  
            The current network.  
        target_net : nn.Module  
            The target network to be updated.  
        """  
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)
        
    def update(self, transition_dict):
        """  
        Update the agent's networks using a batch of transitions.  
  
        Parameters  
        ----------  
        transition_dict : dict  
            Dictionary containing the transitions. Should have keys:  
            'states', 'actions', 'rewards', 'next_states', 'dones'.  
        """  
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        td_target = self.calc_target(rewards, next_states, dones)
        
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)