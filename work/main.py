#!/usr/bin/env python
# coding: utf-8

import numpy as np

## Policy Evaluation
def bellman_update(env, V, pi, s, gamma): 
    sum_v = 0.0
    for a in env.A:
        transitions = env.transitions(s, a)
        for next_state in env.S:
            reward = transitions[next_state, 0]
            transition_prob = transitions[next_state, 1]
            sum_v += pi[s][a] * transition_prob * (reward + gamma * V[next_state])
    V[s] = sum_v

def evaluate_policy(env, V, pi, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))

    return V


## Policy Iteration
def q_greedify_policy(env, V, pi, s, gamma):
    q_pi = np.zeros(len(env.A))
    for a in env.A:
        transitions = env.transitions(s, a)
        for next_state in env.S:
            reward = transitions[next_state, 0]
            transition_prob = transitions[next_state, 1]
            q_pi[a] += transition_prob * (reward + gamma * V[next_state])
    best_action = np.argmax(q_pi)
    pi[s] = np.eye(len(env.A))[best_action]


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


## Value Iteration
def bellman_optimality_update(env, V, s, gamma):
    v = np.zeros(len(env.A))
    
    for a in env.A:
        transitions = env.transitions(s, a)
        for next_state in env.S:
            reward = transitions[next_state, 0]
            transition_prob = transitions[next_state, 1]
            v[a] += transition_prob * (reward + gamma * V[next_state])
    V[s] = np.max(v)

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





## Tests

import gridworld as grid
#import lib.tools as tools

my_grid = grid.GridWorld(4)

#print(my_grid.transitions(0, 3))
#print(my_grid.transitions(1, 3))
#print(my_grid.transitions(2, 3))
#print(my_grid.transitions(3, 3))
gamma = 0.9
theta = 0.1

V, pi = value_iteration(my_grid, gamma, theta)

grid.plot(V, pi)
