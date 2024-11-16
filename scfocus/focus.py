from .environment import Env, ReplayBuffer, train_off_policy
from .model import SAC
import numpy as np
import torch
import tqdm
import time
from scipy.stats import multivariate_normal
from sklearn.preprocessing import minmax_scale
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  

class focus:  
    """  
    Focus class for performing advanced reinforcement learning-based analysis on single-cell data.  

    This class utilizes the Soft Actor-Critic (SAC) reinforcement learning framework to enhance  
    cell subtype discrimination and identify distinct lineage branches within single-cell data.  
    It manages the environment, memory buffers, and the ensemble of SAC agents to train and evaluate  
    the model over multiple episodes.  

    Parameters  
    ----------  
    f : array-like  
        Latent space of the original data, with shape (num_samples, num_features).  
    hidden_dim : int, optional  
        Number of hidden units in the neural networks, by default 128.  
    n : int, optional  
        Number of agents or parallel environments, by default 8.  
    max_steps : int, optional  
        Maximum number of steps per episode, by default 5.  
    pct_samples : float, optional  
        Percentage of samples to use for each state, by default 0.125.  
    n_states : int, optional  
        Number of state variables, by default 2.  
    err_scale : float, optional  
        Error scaling factor for reward calculation, by default 1.  
    bins : int, optional  
        Number of bins for histogram-based state discretization, by default 15.  
    capacity : float, optional  
        Capacity of the replay buffer, by default 1e4.  
    actor_lr : float, optional  
        Learning rate for the actor network, by default 1e-4.  
    critic_lr : float, optional  
        Learning rate for the critic network, by default 1e-3.  
    alpha_lr : float, optional  
        Learning rate for the entropy coefficient, by default 1e-4.  
    target_entropy : float, optional  
        Target entropy for the SAC algorithm, by default -1.  
    tau : float, optional  
        Soft update coefficient for target networks, by default 5e-3.  
    gamma : float, optional  
        Discount factor for future rewards, by default 0.99.  
    num_episodes : int, optional  
        Number of training episodes, by default 1000.  
    batch_size : int, optional  
        Batch size for training, by default 64.  
    res : float, optional  
        Resolution parameter for merging focus patterns, by default 0.05.  
    device : torch.device, optional  
        Device to run the computations on (e.g., CPU or GPU), by default torch.device('cpu').  
    """  

    def __init__(self, f, hidden_dim=128, n=8, max_steps=5, pct_samples=.125, n_states=2,   
                 err_scale=1, bins=15, capacity=1e4, actor_lr=1e-4, critic_lr=1e-3,   
                 alpha_lr=1e-4, target_entropy=-1, tau=5e-3, gamma=.99,   
                 num_episodes=1e3, batch_size=64, res=.05, device=torch.device('cpu')):  
        """  
        Initialize the Focus class.  

        Parameters  
        ----------  
        f : array-like  
            Latent space of the original data, with shape (num_samples, num_features).  
        hidden_dim : int, optional  
            Number of hidden units in the neural networks, by default 128.  
        n : int, optional  
            Number of agents or parallel environments, by default 8.  
        max_steps : int, optional  
            Maximum number of steps per episode, by default 5.  
        pct_samples : float, optional  
            Percentage of samples to use for each state, by default 0.125.  
        n_states : int, optional  
            Number of state variables, by default 2.  
        err_scale : float, optional  
            Error scaling factor for reward calculation, by default 1.  
        bins : int, optional  
            Number of bins for histogram-based state discretization, by default 15.  
        capacity : float, optional  
            Capacity of the replay buffer, by default 1e4.  
        actor_lr : float, optional  
            Learning rate for the actor network, by default 1e-4.  
        critic_lr : float, optional  
            Learning rate for the critic network, by default 1e-3.  
        alpha_lr : float, optional  
            Learning rate for the entropy coefficient, by default 1e-4.  
        target_entropy : float, optional  
            Target entropy for the SAC algorithm, by default -1.  
        tau : float, optional  
            Soft update coefficient for target networks, by default 5e-3.  
        gamma : float, optional  
            Discount factor for future rewards, by default 0.99.  
        num_episodes : int, optional  
            Number of training episodes, by default 1000.  
        batch_size : int, optional  
            Batch size for training, by default 64.  
        res : float, optional  
            Resolution parameter for merging focus patterns, by default 0.05.  
        device : torch.device, optional  
            Device to run the computations on (e.g., CPU or GPU), by default torch.device('cpu').  
        """  
        self.state_d        = (2 + bins) * n_states * n  
        self.hidden_dim     = hidden_dim  
        self.action_d       = 2 * n_states * n  
        self.action_space   = (f[:, :n_states].min().item(), f[:, :n_states].max().item())  
        self.actor_lr       = actor_lr  
        self.critic_lr      = critic_lr  
        self.alpha_lr       = alpha_lr  
        self.target_entropy = target_entropy  
        self.tau            = tau  
        self.gamma          = gamma  
        self.device         = device  
        self.capacity       = capacity  
        self.ensemble       = []  
        self.env            = Env(n, f, max_steps, pct_samples, n_states, err_scale, bins)  
        self.memory         = []  
        self.max_steps      = max_steps  
        self.num_episodes   = num_episodes  
        self.minimal_size   = num_episodes / 10 * max_steps  
        self.batch_size     = batch_size  
        self.res            = res  
        self.fp             = []  
        self.r              = []  
        self.e              = []  

    def meta_focusing(self, n):  
        """  
        Perform meta focusing by iteratively fitting the ensemble and refining focus.  

        Parameters  
        ----------  
        n : int  
            Number of meta focusing iterations to perform.  

        Returns  
        -------  
        self : Focus  
            Returns the instance itself for method chaining.  
        """  
        start = time.time()  
        for i in range(n):  
            self.meta_fit()  
            self.focus_fit(10)  
        end = time.time()  
        print(f'Meta focusing time used: {(end - start):.2f} seconds')  
        return self  

    def meta_fit(self):  
        """  
        Perform a single meta fitting step by training a new SAC agent and updating memory.  

        Returns  
        -------  
        self : Focus  
            Returns the instance itself for method chaining.  
        """  
        start = time.time()  
        self.ensemble.append(SAC(  
            self.state_d,  
            self.hidden_dim,  
            self.action_d,  
            self.action_space,  
            self.actor_lr,  
            self.critic_lr,  
            self.alpha_lr,  
            self.target_entropy,  
            self.tau,  
            self.gamma,  
            self.device  
        ))  
        self.memory.append(ReplayBuffer(self.capacity))  
        r, e = train_off_policy(  
            self.env,  
            self.ensemble[-1],  
            self.memory[-1],  
            self.num_episodes,  
            self.minimal_size,  
            self.batch_size  
        )  
        self.r.append(np.vstack(r).ravel())  
        self.e.append(np.vstack(e).ravel())  
        end = time.time()  
        print(f'Meta fitting time used: {(end - start):.2f} seconds')  
        return self  

    def focus_fit(self, episodes):  
        """  
        Fit the focus model over a specified number of episodes.  

        This method iteratively updates the focus weights based on the actions taken by the  
        SAC agent within the environment. It monitors the convergence by checking the  
        change in weights and stops early if the change is below a threshold.  

        Parameters  
        ----------  
        episodes : int  
            Number of episodes to train the focus model.  

        Returns  
        -------  
        self : Focus  
            Returns the instance itself for method chaining.  
        """  
        start = time.time()  
        episode_weight = []  
        with tqdm.tqdm(total=int(episodes), desc='Focus fitting...') as pbar:  
            self.weights = None  
            for i_episode in range(int(episodes)):  
                ls_weights = []  
                state = self.env.reset()  
                for i in range(self.max_steps):  
                    with torch.no_grad():  
                        action = self.ensemble[-1].take_action(state)  
                    action = action.ravel()  
                    mus = action[:int(action.shape[-1]/2)]  
                    logstds = action[int(action.shape[-1]/2):]  
                    L = self.env.n_states  
                    bra_weights = []  
                    for j in range(self.env.n):  
                        mu = mus[L*j:L*(j+1)]  
                        logstd = logstds[L*j:L*(j+1)]  
                        std = np.log1p(np.exp(logstd))  
                        mn = multivariate_normal(mu, np.diag(self.env.sigma / (1 + np.exp(-std))))  
                        weights = minmax_scale(mn.logpdf(self.env.f[:, :self.env.n_states]))  
                        bra_weights.append(weights)  
                    ls_weights.append(bra_weights)  
                    next_state, _, _ = self.env.step(action)  
                    state = next_state  
                weights = np.array(ls_weights)  
                if self.weights is not None:  
                    err = np.linalg.norm(weights - self.weights)  
                    if err < 3 and i_episode > 2:  
                        break  
                self.weights = np.array(ls_weights)  
                episode_weight.append(ls_weights)  
                pbar.update(1)  
        fp = np.array(episode_weight)  
        self.fp.append(fp.T.mean(axis=-1).mean(axis=-1))  
        end = time.time()  
        print(f'Focus fitting time used: {(end - start):.2f} seconds')  
        return self  

    def merge_fp2(self):  
        """  
        Merge focus patterns by performing two levels of merging.  

        This method first calls `merge_fp` to perform initial merging of focus patterns,  
        then concatenates all merged focus patterns into a single array for further processing.  

        Returns  
        -------  
        self : Focus  
            Returns the instance itself for method chaining.  
        """  
        self.merge_fp()  
        self.fp = [np.hstack(self.mfp)]  
        self.merge_fp()  
        return self  

    def merge_fp(self):  
        """  
        Merge focus patterns based on similarity thresholds.  

        This method groups focus patterns that have significant overlap based on the  
        specified resolution parameter. It computes the mean of grouped focus patterns  
        to create merged focus patterns.  

        Returns  
        -------  
        self : Focus  
            Returns the instance itself after merging focus patterns.  
        """  
        self.mfp = []  
        for fp in self.fp:  
            n = int(fp.shape[0] * self.res)  
            ord_indices = np.argsort(fp, axis=0)[-n:, :]  
            groups = []  
            for i in range(fp.shape[1]):  
                if any([i in g for g in groups]):  
                    continue  
                g_ = [i]  
                if i != fp.shape[1] - 1:  
                    for j in range(i + 1, fp.shape[1]):  
                        if len(set(ord_indices[:, i]).intersection(set(ord_indices[:, j]))) > 0.25 * n:  
                            g_.append(j)  
                    groups.append(g_)  
                else:  
                    groups.append(g_)  
            mfp = []  
            for g in groups:  
                if len(g) > 1:  
                    mfp.append(fp[:, g].mean(axis=1)[:, np.newaxis])  
                else:  
                    mfp.append(fp[:, g])  
            mfp = np.hstack(mfp)  
            self.mfp.append(mfp)  
        return self  

    def focus_diff(self):  
        """  
        Calculate entropy and pseudotime based on merged focus patterns.  

        This method computes the entropy of the merged focus patterns and scales it to  
        derive pseudotime values, which can be used for further analysis of cell differentiation trajectories.  

        Returns  
        -------  
        self : Focus  
            Returns the instance itself with updated entropy and pseudotime attributes.  
        """  
        self.entropy = (self.mfp * -np.log(self.mfp)).sum(axis=1)  
        self.pseudotime = 1 - minmax_scale(self.entropy)  
        return self
        
    
        
