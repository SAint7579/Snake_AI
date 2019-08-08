from Snake import Game as game
import pygame
from pygame.locals import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
env = game()
env.reset()
action = -1
import random

import keras
from statistics import median, mean
from collections import Counter
import numpy as np

LR = 0.0001
goal_steps = 300
score_requirement = 50
initial_games = 3000

global_steps = 300


def generate_train_set(model):
    '''
        To generate training set for the model
    '''
    global score_requirement
    training_collected = 0
    training_data = []
    scores = []
    accepted_scores = []
    while(training_collected == 0):
        print("Score Requirement: ", score_requirement)
        for _ in range(initial_games):
            print("Game ", _, "out of", str(initial_games), '\r', end='')
            #To start again
            env.reset()
            score = 0
            #Moves executed in the current environment
            game_memory = []
            #Observation in this environment
            prev_observation = []

            for _ in range(goal_steps):
                if len(prev_observation) == 0:
                    action = random.randrange(0,3)
                else:
                    if not model:
                        #If no model exists
                        action = random.randrange(0,3)
                    else:
                        #Taking output form the model
                        prediction = model.predict(prev_observation.reshape(-1, len(prev_observation), 1))
                        action = np.argmax(prediction[0])
                        
                #Taking the new observation after the steps
                observation, reward, done, info = env.step(action)
                if len(prev_observation) > 0:
                    game_memory.append([prev_observation, action])
                #last observation matrix
                prev_observation = observation
                #Incrementing rewards
                score += reward
                if done:
                    break

            #Saving the modle if the score is higher than our threshold/required socre
            if score >= score_requirement:
                #Setting flag to 1
                training_collected = 1
                accepted_scores.append(score)
                for data in game_memory:
                    action_sample = [0,0,0]
                    action_sample[data[1]] = 1
                    output = action_sample
                    # Collecting the training data
                    training_data.append(np.array([data[0],output]))
            scores.append(score)

                
    # Printing the stats
    print('Average accepted score:', mean(accepted_scores))
    print('Score Requirement:', score_requirement)
    print('Median score for accepted scores:', median(accepted_scores))
    #score_requirement = mean(accepted_scores)
    # Saving the stats
    training_data_save = np.array([training_data, score_requirement])
    np.save('saved.npy', training_data_save)
 
    return training_data
         
def create_dummy_models(training_data):
    '''
        To initialize the DNN model
    '''
    shape_second_parameter = len(training_data[0][0])
    x = np.array([i[0] for i in training_data])
    X = x.reshape(-1,shape_second_parameter)
    y = [i[1] for i in training_data]
    model = create_neural_network_model(input_size=len(X[0]), output_size=len(y[0]))
    return model

def create_neural_network_model(input_size, output_size):
    '''
        Simple 2 layer DNN
    '''
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=64,input_dim=input_size,name= 'hidden1'))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(units=32,name = 'hidden2'))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(units=output_size,name = 'output'))
    model.add(keras.layers.Activation("softmax"))
    return model

def train_model(training_data,model = False):
    shape_second_parameter = len(training_data[0][0])
    x = np.array([i[0] for i in training_data])
    X = x.reshape(-1, shape_second_parameter)
    y = np.array([np.array(i[1]) for i in training_data])
    print("Training the Model")
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=10, batch_size=16, verbose = 1)
    model.save('miniskake_trained.tflearn')
 
    return model
def evaluate(model):
    # now it's time to evaluate the trained model
    scores = []
    choices = []
    for each_game in range(5):
        score = 0
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0, 3)
            else:
                prediction = model.predict(prev_obs.reshape(-1, len(prev_obs)))
                action = np.argmax(prediction[0])
 
            choices.append(action)
 
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            score += reward
            if done: break
 
        scores.append(score)
    print('Average Score:', sum(scores) / len(scores))

            
if __name__ == "__main__":
 
    training_data = generate_train_set(None)
    model = create_dummy_models(training_data)
    model = train_model(training_data, model)
    evaluate(model)
        # recursive learning
    generation = 1
    while True:
        generation += 1
        score_requirement = int(input("Enter the score requirement: "))
        print('Generation: ', generation)
        # training_data = initial_population(model)
        training_data = np.append(training_data, generate_train_set(None), axis=0)
        print('generation: ', generation, ' initial population: ', len(training_data))
        if len(training_data) == 0:
            break
        model = train_model(training_data, model)
        evaluate(model)
