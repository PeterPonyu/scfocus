import random
import collections
import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from scipy.stats import multivariate_normal

class Env:
    """  
    Environment class for simulating and managing states, actions, and rewards.  

    Parameters  
    ----------  
    n : int  
        Number of individuals in the environment.  
    f : numpy.ndarray  
        Input feature matrix with shape (num_samples, num_features).  
    max_steps : int  
        Maximum number of steps before the environment terminates.  
    pct_samples : float  
        Percentage of samples to be used for each state.  
    n_states : int  
        Number of state variables.  
    err_scale : float  
        Error scaling factor to adjust the weight of errors in reward calculation.  
    bins : int  
        Number of bins for histogram-based state discretization.  
    """  
    def __init__(self, n, f, max_steps, pct_samples, n_states, err_scale, bins):
        
        self.n                   = n
        self.f                   = f
        self.max_steps           = max_steps
        self.n_samples           = int(f.shape[0]*pct_samples)
        self.n_states            = n_states
        self.err_scale           = err_scale
        self.bins                = bins
        self.sigma               = f[:, :n_states].std(axis=0).max()
        self.value_list          = []
        

    def reset(self):
        """  
        Resets the environment to its initial state.  

        Returns  
        -------  
        state : numpy.ndarray  
            Initial state with shape (1, state_dimension).  
        """  
        self.cnt = 0
        states = []
        errs = []
        for i in range(self.n):
            idx = pd.Series(np.arange(self.f.shape[0])).sample(self.n_samples).to_list()
            f_  = self.f[idx, :]
            state, err = self.get_state(f_)
            states.append(state)
            errs.append(err)
        normall = self.get_norm(states)
        err = np.array(errs).mean()
        state = np.hstack(states)[np.newaxis, :]
        self.err = err
        return state
    
    def step(self, action):
        """  
        Executes an action and updates the environment's state.  

        Parameters  
        ----------  
        action : numpy.ndarray  
            Action vector with shape (action_dimension,).  

        Returns  
        -------  
        state : numpy.ndarray  
            Updated state with shape (1, state_dimension).  
        reward : numpy.ndarray  
            Reward value with shape (1, 1).  
        done : numpy.ndarray  
            Boolean flag indicating if the environment has terminated, shape (1, 1).  
        """ 
        self.cnt += 1
        action = action.ravel()
        mus = action[:int(action.shape[-1]/2)]
        logstds = action[int(action.shape[-1]/2):]
        L = self.n_states
        states = []
        errs = []
        for i in range(self.n):
            mu = mus[L*i:L*(i+1)]
            logstd = logstds[L*i:L*(i+1)]
            std = np.log1p(np.exp(logstd))
            mn = multivariate_normal(mu, np.diag(self.sigma / (1 + np.exp(-std))))
            weights = minmax_scale(mn.logpdf(self.f[:, :self.n_states]))
            idx = pd.Series(np.arange(self.f.shape[0], dtype=int)).sample(self.n_samples, weights=weights).to_list()
            f_ = self.f[idx, :]
            state, err = self.get_state(f_)
            states.append(state)
            errs.append(err)
        normall = self.get_norm(states)
        err = np.array(errs).mean()
        state = np.hstack(states)[np.newaxis, :]
        reward = np.array([normall - err * self.err_scale])[np.newaxis, :]
        done = np.array([True] if self.cnt >= self.max_steps else [False])[np.newaxis, :]
        self.err = err
        self.normall = normall
        self.value_list.append((reward, normall, err))
        return state, reward, done

    def get_norm(self, states):
        """  
        Calculates the average norm distance between all pairs of states.  

        Parameters  
        ----------  
        states : list of numpy.ndarray  
            List of state vectors.  

        Returns  
        -------  
        normall : float  
            Average norm distance between states.  
        """ 
        norms = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            for j in range(i + 1, self.n - 1):
                norm = np.linalg.norm(states[i] - states[j])
                norms[i, j] = norm
                norms[j, i] = norm
        normall = norms.sum() / (self.n*(self.n-1))
        return normall
    
    def get_state(self, f_):
        """  
        Computes the current state and error based on input features.  

        Parameters  
        ----------  
        f_ : numpy.ndarray  
            Subset of input features with shape (num_samples, n_states).  

        Returns  
        -------  
        state : numpy.ndarray  
            Current state vector containing normalized histogram bins, means, and standard deviations.  
        err : float  
            Error value of the current state.  
        """  
        state_bins_ls = []
        for i in range(self.n_states):
            state_bins = minmax_scale(np.histogram(f_[:, i], bins=self.bins)[0])
            state_bins_ls.append(state_bins)
        f_bins = np.hstack(state_bins_ls)
        state  = np.hstack([f_bins,
                           f_[:, :self.n_states].mean(axis=0),
                           f_[:, :self.n_states].std(axis=0)
                           ])
        err    = sum([f_[:, i].std() for i in range(self.n_states)]) / self.n_states
        return state, err
    
class ReplayBuffer:
   
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=int(capacity))
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.vstack(state), np.vstack(action), np.vstack(reward), np.vstack(next_state), np.vstack(done)
    
    def size(self):
        return len(self.buffer)
    
def train_off_policy(env, agent, replay_buffer, num_episodes, minimal_size, batch_size):
   
    return_list = []
    err_list    = []
    one_episode = int(num_episodes/10)
    for i in range(10):
        with tqdm.tqdm(total=one_episode, desc='Meta fitting... %d'%(i+1)) as pbar:
            for i_episode in range(one_episode):
                state          = env.reset()
                episode_return = 0
                episode_err    = env.err
                done           = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    episode_err += env.err
                return_list.append(episode_return)
                err_list.append(episode_err)
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states':b_s, 'actions':b_a, 'next_states':b_ns, 'rewards':b_r,'dones':b_d}
                    agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'E':'%d' % (one_episode*i+i_episode+1),
                                     'R':'%.2f'%np.mean(return_list[-10:]),
                                     'S':'%.2f'%np.mean(err_list[-10:])
                                     })
                pbar.update(1)
            if i > 5 and .01*np.array(err_list[-one_episode:]).mean() > np.array(err_list[-one_episode:]).std():
                print(f'Converged at iteration {i+1}. Training stopped!')
                break
    return return_list, err_list




