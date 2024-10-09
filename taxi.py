import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3",render_mode="ansi") #init environment

episodes = 10000 #num of episodes
alpha = 0.1 #learning rate
gamma = 0.7 #discount factor
epsilon = 0.01 #exploration rate
interarctions = 100 #max interactions
 
#init Q table
Q = np.zeros((env.observation_space.n, env.action_space.n))

training_rewards = []

#LEARN
for episode in range(episodes):

    print(f"episode number {episode}") #tracking the training
    total_rewards = 0
    state, _ = env.reset()

    for interact in range(interarctions):  
        
        if np.random.uniform(0, 1) < epsilon:   #exploitation or exploration
            action = np.random.randint(0,env.action_space.n)
        else:
            action = np.argmax(Q[state,:])
            
        new_state, reward, done, truncated, info = env.step(action) #observing

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action]) #Updating Q learning table

        state = new_state #update state
        total_rewards += reward

        if done: 
            break
    
    training_rewards.append(total_rewards)

#Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(training_rewards)
plt.title('Learning Curve: Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
    
#EVALUATION
eval_episodes = 10
total_rewards_eval = []
total_steps = []

for episode in range(eval_episodes):
    state, _ = env.reset()
    total_rewards = 0
    steps = 0
    print(f"Evaluating {episode}") #tracking the evaluation

    while True:
        action = np.argmax(Q[state, :]) #Always choose the highest value from Q table
        new_state, reward, done, truncated, info = env.step(action)

        total_rewards += reward
        state = new_state
        steps += 1
        print(steps)
        if done:
            break
    
    total_rewards_eval.append(total_rewards)
    total_steps.append(steps)

avg_reward = np.mean(total_rewards_eval)
avg_steps = np.mean(total_steps)

print(f"Average Total Reward during Evaluation: {avg_reward}")
print(f"Average Number of Steps during Evaluation: {avg_steps}")
        