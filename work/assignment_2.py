#!/usr/bin/env python
# coding: utf-8

import numpy as np
import lib.tools as tools
import lib.grader as grader

## Section 1: Policy Evaluation
# 
# $$\large v(s) \leftarrow \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
# 

def evaluate_policy(env, V, pi, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
            
    return V

def bellman_update(env, V, pi, s, gamma): 
    sum_v = 0.0
    for a in env.A:
        transitions = env.transitions(s, a)
        for next_state in env.S:
            reward = transitions[next_state, 0]
            transition_prob = transitions[next_state, 1]
            sum_v += pi[s][a] * transition_prob * (reward + gamma * V[next_state])
    V[s] = sum_v

# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

# set up test environment
num_spaces = 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)

# build test policy
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1

gamma = 0.9
theta = 0.1

V = np.zeros(num_spaces + 1)
V = evaluate_policy(env, V, city_policy, gamma, theta)

print(V)

# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.

# set up test environment
num_spaces = 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)

# build test policy
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1

gamma = 0.9
theta = 0.1

V = np.zeros(num_spaces + 1)
V = evaluate_policy(env, V, city_policy, gamma, theta)

# test the value function
answer = [80.04, 81.65, 83.37, 85.12, 86.87, 88.55, 90.14, 91.58, 92.81, 93.78, 87.77]

# make sure the value function is within 2 decimal places of the correct answer
assert grader.near(V, answer, 1e-2)


# You can use the ``plot`` function to visualize the final value function and policy.


tools.plot(V, city_policy)


# Observe that the value function qualitatively resembles the city council's preferences &mdash; it monotonically increases as more parking is used, until there is no parking left, in which case the value is lower. Because of the relatively simple reward function (more reward is accrued when many but not all parking spots are taken and less reward is accrued when few or all parking spots are taken) and the highly stochastic dynamics function (each state has positive probability of being reached each time step) the value functions of most policies will qualitatively resemble this graph. However, depending on the intelligence of the policy, the scale of the graph will differ. In other words, better policies will increase the expected return at every state rather than changing the relative desirability of the states. Intuitively, the value of a less desirable state can be increased by making it less likely to remain in a less desirable state. Similarly, the value of a more desirable state can be increased by making it more likely to remain in a more desirable state. That is to say, good policies are policies that spend more time in desirable states and less time in undesirable states. As we will see in this assignment, such a steady state distribution is achieved by setting the price to be low in low occupancy states (so that the occupancy will increase) and setting the price high when occupancy is high (so that full occupancy will be avoided).

# ## Section 2: Policy Iteration
# Now the city council would like you to compute a more efficient policy using policy iteration. Policy iteration works by alternating between evaluating the existing policy and making the policy greedy with respect to the existing value function. We have written an outline of the policy iteration algorithm described in chapter 4.3 of the textbook. We will make use of the policy evaluation algorithm you completed in section 1. It is left to you to fill in the `q_greedify_policy` function, such that it modifies the policy at $s$ to be greedy with respect to the q-values at $s$, to complete the policy improvement algorithm.

# In[18]:


def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        
        if not np.array_equal(pi[s], old):
            policy_stable = False
            
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
        
    return V, pi


# In[23]:


# -----------
# Graded Cell
# -----------
def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    # YOUR CODE HERE    
    my_A = np.zeros(len(env.A))
    for a in env.A:
        transitions = env.transitions(s, a)
        for s_prime in env.S:
            reward = transitions[s_prime, 0]
            prob_s_prime_given_s_a = transitions[s_prime, 1]
            my_A[a] += prob_s_prime_given_s_a * (reward + gamma * V[s_prime])
    best_action = np.argmax(my_A)
    pi[s] = np.eye(len(env.A))[best_action]


# In[24]:


# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

gamma = 0.9
theta = 0.1
env = tools.ParkingWorld(num_spaces=6, num_prices=4)

V = np.array([7, 6, 5, 4, 3, 2, 1])
pi = np.ones((7, 4)) / 4

new_pi, stable = improve_policy(env, V, pi, gamma)

# expect first call to greedify policy
expected_pi = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])
assert np.all(new_pi == expected_pi)
assert stable == False

# the value function has not changed, so the greedy policy should not change
new_pi, stable = improve_policy(env, V, new_pi, gamma)

assert np.all(new_pi == expected_pi)
assert stable == True


# In[25]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.
gamma = 0.9
theta = 0.1
env = tools.ParkingWorld(num_spaces=10, num_prices=4)

V, pi = policy_iteration(env, gamma, theta)

V_answer = [81.60, 83.28, 85.03, 86.79, 88.51, 90.16, 91.70, 93.08, 94.25, 95.25, 89.45]
pi_answer = [
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

# make sure value function is within 2 decimal places of answer
assert grader.near(V, V_answer, 1e-2)
# make sure policy is exactly correct
assert np.all(pi == pi_answer)


# When you are ready to test the policy iteration algorithm, run the cell below.

# In[26]:


env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = policy_iteration(env, gamma, theta)


# You can use the ``plot`` function to visualize the final value function and policy.

# In[27]:


tools.plot(V, pi)


# You can check the value function (rounded to one decimal place) and policy against the answer below:<br>
# State $\quad\quad$    Value $\quad\quad$ Action<br>
# 0 $\quad\quad\quad\;$        81.6 $\quad\quad\;$ 0<br>
# 1 $\quad\quad\quad\;$        83.3 $\quad\quad\;$ 0<br>
# 2 $\quad\quad\quad\;$        85.0 $\quad\quad\;$ 0<br>
# 3 $\quad\quad\quad\;$        86.8 $\quad\quad\;$ 0<br>
# 4 $\quad\quad\quad\;$        88.5 $\quad\quad\;$ 0<br>
# 5 $\quad\quad\quad\;$        90.2 $\quad\quad\;$ 0<br>
# 6 $\quad\quad\quad\;$        91.7 $\quad\quad\;$ 0<br>
# 7 $\quad\quad\quad\;$        93.1 $\quad\quad\;$ 0<br>
# 8 $\quad\quad\quad\;$        94.3 $\quad\quad\;$ 0<br>
# 9 $\quad\quad\quad\;$        95.3 $\quad\quad\;$ 3<br>
# 10 $\quad\quad\;\;\,\,$      89.5 $\quad\quad\;$ 3<br>

# ## Section 3: Value Iteration
# The city has also heard about value iteration and would like you to implement it. Value iteration works by iteratively applying the Bellman optimality equation for $v_{\ast}$ to a working value function, as an update rule, as shown below.
# 
# $$\large v(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
# We have written an outline of the value iteration algorithm described in chapter 4.4 of the textbook. It is left to you to fill in the `bellman_optimality_update` function to complete the value iteration algorithm.

# In[28]:


def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi


# In[29]:


# -----------
# Graded Cell
# -----------
def bellman_optimality_update(env, V, s, gamma):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    # YOUR CODE HERE
    v = np.zeros(len(env.A))
    for a in env.A:
        transitions = env.transitions(s, a)
        for s_prime in env.S:
            reward = transitions[s_prime, 0]
            prob_s_prime_given_s_a = transitions[s_prime, 1]
            v[a] += prob_s_prime_given_s_a * (reward + gamma * V[s_prime])
    V[s] = np.max(v)


# In[30]:


# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

gamma = 0.9
env = tools.ParkingWorld(num_spaces=6, num_prices=4)

V = np.array([7, 6, 5, 4, 3, 2, 1])

# only state 0 updated
bellman_optimality_update(env, V, 0, gamma)
assert list(V) == [5, 6, 5, 4, 3, 2, 1]

# only state 2 updated
bellman_optimality_update(env, V, 2, gamma)
assert list(V) == [5, 6, 7, 4, 3, 2, 1]


# In[31]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.
gamma = 0.9
env = tools.ParkingWorld(num_spaces=10, num_prices=4)

V = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

for _ in range(10):
    for s in env.S:
        bellman_optimality_update(env, V, s, gamma)

# make sure value function is exactly correct
answer = [61, 63, 65, 67, 69, 71, 72, 74, 75, 76, 71]
assert np.all(V == answer)


# When you are ready to test the value iteration algorithm, run the cell below.

# In[32]:


env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration(env, gamma, theta)


# You can use the ``plot`` function to visualize the final value function and policy.

# In[33]:


tools.plot(V, pi)


# You can check your value function (rounded to one decimal place) and policy against the answer below:<br>
# State $\quad\quad$    Value $\quad\quad$ Action<br>
# 0 $\quad\quad\quad\;$        81.6 $\quad\quad\;$ 0<br>
# 1 $\quad\quad\quad\;$        83.3 $\quad\quad\;$ 0<br>
# 2 $\quad\quad\quad\;$        85.0 $\quad\quad\;$ 0<br>
# 3 $\quad\quad\quad\;$        86.8 $\quad\quad\;$ 0<br>
# 4 $\quad\quad\quad\;$        88.5 $\quad\quad\;$ 0<br>
# 5 $\quad\quad\quad\;$        90.2 $\quad\quad\;$ 0<br>
# 6 $\quad\quad\quad\;$        91.7 $\quad\quad\;$ 0<br>
# 7 $\quad\quad\quad\;$        93.1 $\quad\quad\;$ 0<br>
# 8 $\quad\quad\quad\;$        94.3 $\quad\quad\;$ 0<br>
# 9 $\quad\quad\quad\;$        95.3 $\quad\quad\;$ 3<br>
# 10 $\quad\quad\;\;\,\,$      89.5 $\quad\quad\;$ 3<br>

# In the value iteration algorithm above, a policy is not explicitly maintained until the value function has converged. Below, we have written an identically behaving value iteration algorithm that maintains an updated policy. Writing value iteration in this form makes its relationship to policy iteration more evident. Policy iteration alternates between doing complete greedifications and complete evaluations. On the other hand, value iteration alternates between doing local greedifications and local evaluations. 

# In[34]:


def value_iteration2(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            q_greedify_policy(env, V, pi, s, gamma)
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, pi


# You can try the second value iteration algorithm by running the cell below.

# In[35]:


env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration2(env, gamma, theta)
tools.plot(V, pi)


# ## Wrapping Up
# Congratulations, you've completed assignment 2! In this assignment, we investigated policy evaluation and policy improvement, policy iteration and value iteration, and Bellman updates. Gridworld City thanks you for your service!
