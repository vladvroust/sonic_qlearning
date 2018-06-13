from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from collections import deque
from keras.callbacks import TensorBoard
from skimage import color
from skimage.transform import resize
import gym_remote.exceptions as gre
import os
import random
import numpy as np
import gym
import retro
import copy
from retro_contest.local import make
import time
import tensorflow as tf
from datetime import timedelta
from sklearn.metrics import mean_squared_error

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

#env specific
class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], ['UP']] #adding UP
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

#end of env specific

def main():
    
    start_time = time.time()

    games = ["SonicTheHedgehog-Genesis","SonicTheHedgehog2-Genesis","SonicAndKnuckles3-Genesis"]
    game = np.random.choice(games,1)[0]
    state = np.random.choice(retro.list_states(game),1)[0]
    #env = retro.make(game, state)
    env = AllowBacktracking(make(game, state)) #contest version
    env = SonicDiscretizer(env) #contest version
    print(game,'-',state)

    # Parameters
    timesteps = 4500
    memory = deque(maxlen=30000)
    epsilon = 1                                #probability of doing a random move
    max_random = 1
    min_random = 0.1                           #minimun randomness #r12
    rand_decay = 1e-3                                #reduce the randomness by decay/loops
    gamma = 0.99                               #discount for future reward
    mb_size = 256                               #learning minibatch size
    loops = 45                               #loop through the different game levels
    sub_loops = 100
    hold_action = 1                            #nb frames during which we hold (4 for normal, 1 for contest)
    learning_rate = 5e-5
    max_reward = 0
    min_reward = 10000
    #action_threshold = 1
    target_step_interval = 10
    reward_clip = 200
    resize_to = [128,128]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        #input observation state, output Q of actions
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8,8), strides = 4, activation="relu", input_shape=(128,128,3)))
        model.add(Conv2D(64, kernel_size=(4,4), strides = 2, activation="relu"))
        model.add(Conv2D(64, (3,3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(env.action_space.n, kernel_initializer="uniform", activation="linear"))

        if os.path.isfile("sonic_model.h5"):
            model.load_weights("sonic_model.h5")

        model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

        tensorboard = TensorBoard(log_dir="logs/sonic_modmemdecayrdq18_reshape_64x512mb256_resc_target_interval_{}_memory_30000_lr_{}_decay_{}.{}".format(target_step_interval,learning_rate, rand_decay, time.time()))
        tensorboard.set_model(model)
        train_names = ["Loss", "Accuracy"]

        # serialize model to JSON
        model_json = model.to_json()
        with open("sonic_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("sonic_model.h5")
        model.save_weights("sonic_target_model.h5")

        env.close()

    for training_loop in range(loops):
        with tf.Session(config=config) as sess:
            model = model_from_json(model_json)
            model.load_weights("sonic_model.h5")
            model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

            target_model = model_from_json(model_json)
            target_model.load_weights("sonic_target_model.h5")
            target_model.trainable = False
            target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

            for sub_training_loop in range(sub_loops):
                loop_start_time = time.time()
                game = np.random.choice(games,1)[0]
                state = np.random.choice(retro.list_states(game),1)[0]
                print("Playing",game,"-",state)
                #env = AllowBacktracking(retro.make(game, state))
                env = AllowBacktracking(make(game, state)) #contest version
                env = SonicDiscretizer(env) #contest version
                obs = env.reset() #game start
                obs_resized = resize(obs, resize_to)
                diff_obs = obs_resized #difference between obs_new and obs to capture velocity

                done = False
                total_raw_reward = 0.0

                #Observation
                for t in range(timesteps):
                    #env.render() #display training
                    if np.random.rand() <= epsilon:
                        #pick a random action
                        action = env.action_space.sample()                    

                        reward_hold = np.zeros(hold_action)
                        for h in range(hold_action):
                            obs_new, reward_hold[h], done, info = env.step(action)     # result of action
                        reward = sum(reward_hold)
                        reward_ = min(reward,reward_clip)
                        
                        obs_new_resized = resize(obs_new, resize_to)
                        diff_obs_new = obs_new_resized - obs_resized

                        #Bellman double Q
                        Q = model.predict(diff_obs[np.newaxis,:])          # Q-values predictions

                        Q_ = model.predict(diff_obs_new[np.newaxis,:])
                        Q_target = target_model.predict(diff_obs_new[np.newaxis,:])

                        target_ = copy.copy(Q)

                        if done:
                            target_[0,action] = reward_ - reward_clip
                        else:
                            target_[0,action] = reward_ + gamma * Q_target[0,:][np.argmax(Q_[0,:])]

                        distance_from_target = mean_squared_error(Q, target_)
                    else:
                        Q = model.predict(diff_obs[np.newaxis,:])          # Q-values predictions

                        action = np.argmax(Q)

                        reward_hold = np.zeros(hold_action)
                        for h in range(hold_action):
                            obs_new, reward_hold[h], done, info = env.step(action)     # result of action
                        reward = sum(reward_hold)
                        reward_ = min(reward,reward_clip)
                        
                        obs_new_resized = resize(obs_new, resize_to)
                        diff_obs_new = obs_new_resized - obs_resized
                        
                        #Bellman double Q
                        Q_ = model.predict(diff_obs_new[np.newaxis,:])
                        Q_target = target_model.predict(diff_obs_new[np.newaxis,:])

                        target_ = copy.copy(Q)

                        if done:
                            target_[0,action] = reward_ - reward_clip
                        else:
                            target_[0,action] = reward_ + gamma * Q_target[0,:][np.argmax(Q_[0,:])]

                        distance_from_target = mean_squared_error(Q, target_)
                        #print("distance from target",distance_from_target)

                    total_raw_reward += reward

                    #memory.append((diff_obs, action, reward, diff_obs_new, done))
                    max_reward = max(reward, max_reward)
                    min_reward = min(reward, min_reward)

                    if distance_from_target > 25:
                        memory.append((diff_obs, action, reward, diff_obs_new, done))
                    elif done:
                        memory.append((diff_obs, action, reward, diff_obs_new, done))

                    # save obs state
                    obs_resized = obs_new_resized
                    diff_obs = diff_obs_new

                    if done:
                        obs = env.reset()           #restart game if done
                        obs_resized = resize(obs, resize_to)
                        diff_obs = obs_resized #difference between obs_new and obs to capture velocity
                        
                epsilon = min_random + (max_random-min_random)*np.exp(-rand_decay*(training_loop*sub_loops + sub_training_loop+1))
                print("Total reward: {}".format(round(total_raw_reward)))
                print("Observation Finished",sub_training_loop+1,"x",training_loop+1,"out of",sub_loops,"x",loops)

                # Learning
                if len(memory) >= mb_size:
                    minibatch_train_start_time = time.time()

                    #sample memory
                    minibatch = random.sample(memory, mb_size)

                    inputs_shape = (mb_size,) + obs_resized.shape
                    inputs = np.zeros(inputs_shape)
                    targets = np.zeros((mb_size, env.action_space.n))

                    #double Q: fix the target for a time to stabilise the model
                    if (sub_training_loop+1)%target_step_interval == 0:# and training_loop*sub_loops + sub_training_loop+1 >= 100: #r12 chase score first
                        #with tf.device('/cpu:0'):
                        # serialize weights to HDF5
                        #model.save_weights("sonic_model.h5")
                        model.save_weights("sonic_target_model.h5")
                        if training_loop*sub_loops + sub_training_loop+1 == 500:
                            model.save_weights("sonic_model_500.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 1000:
                            model.save_weights("sonic_model_1000.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 1500:
                            model.save_weights("sonic_model_1500.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 2000:
                            model.save_weights("sonic_model_2000.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 2500:
                            model.save_weights("sonic_model_2500.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 3000:
                            model.save_weights("sonic_model_3000.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 3500:
                            model.save_weights("sonic_model_3500.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 4000:
                            model.save_weights("sonic_model_4000.h5")
                        elif training_loop*sub_loops + sub_training_loop+1 == 4500:
                            model.save_weights("sonic_model_4500.h5")
                        # create model from json and do inference on cpu
                        target_model = model_from_json(model_json)
                        target_model.load_weights("sonic_target_model.h5")
                        target_model.trainable = False
                        target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

                    for i in range(0, mb_size):
                        diff_obs = minibatch[i][0]
                        action = minibatch[i][1]
                        reward = minibatch[i][2]
                        diff_obs_new = minibatch[i][3]
                        done = minibatch[i][4]

                        #reward clipping
                        reward = min(reward,reward_clip)

                        #Bellman double Q
                        inputs[i] = diff_obs[np.newaxis,:]
                        Q = model.predict(diff_obs[np.newaxis,:])
                        Q_ = model.predict(diff_obs_new[np.newaxis,:])
                        Q_target = target_model.predict(diff_obs_new[np.newaxis,:])

                        targets[i] = copy.copy(Q)

                        if done:
                            targets[i, action] = reward - reward_clip
                        else:
                            targets[i, action] = reward + gamma * Q_target[0,:][np.argmax(Q_[0,:])]

                    #train network on constructed inputs,targets
                    logs = model.train_on_batch(inputs, targets)
                    write_log(tensorboard, train_names, logs, training_loop*sub_loops + sub_training_loop)

                    model.save_weights("sonic_model.h5")

                    print("Model minibatch training lasted:",
                          str(timedelta(seconds=time.time()-minibatch_train_start_time)),"dd:hh:mm:ss")
                    print("Learning Finished",sub_training_loop+1,"x",training_loop+1,"out of",sub_loops,"x",loops)

                env.close()

                print("Loop lasted:",str(timedelta(seconds=time.time()-loop_start_time)),"dd:hh:mm:ss")
                print("Training lasted:",str(timedelta(seconds=time.time()-start_time)),"dd:hh:mm:ss")
                print("Rewards between",min_reward,"and",max_reward)
                print("Pourcentage of random movements set to", epsilon*100, "%\n")

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)

      
