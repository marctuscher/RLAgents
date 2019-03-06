from datetime import datetime
from gym.spaces import Box, Discrete
import numpy as np
import scipy.signal



def discount(rewards, gamma):
    R = 0
    G = []
    for r in rewards[::-1]:
        R = r + gamma * R
        G = [R] + G
    return G



def discount_td(rewards, values, bootstrap, gamma, lam):
    values = np.append(values, bootstrap)
    rewards = np.append(rewards, bootstrap)

    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    # ascontigous and .copy() because torch hates everything else
    advs = np.ascontiguousarray(discount_cumsum(deltas, gamma * lam).copy())
    rtg = np.ascontiguousarray(discount_cumsum(rewards, gamma).copy()[:-1])

    return advs, rtg

def get_cool_looking_datestring():
    now = datetime.now()
    return '' + str(now.day) + '_' + str(now.month) + '_' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)    


def get_space_shape(space):
    if isinstance(space, Box):
        return space.shape[0] if np.isscalar(space.shape) else list(space.shape)
    elif isinstance(space, Discrete):
        return space.n



def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]