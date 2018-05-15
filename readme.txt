Deep Q Learning with Inception Module

Project Team:
    Nishant Bhattacharya	01FB14ECS136
    Saurabh Sood    		    01FB14ECS215
    Srinidhi Sridhar		        01FB14ECS252

Dependancies:

    pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
    
    pip install gym[atari]
    
    pip install tensorflow
    
    pip install keras
    
Dataset:
    Our training dataset is obatained from the gym library. It provides video game environments to train agents. 
    
    These environments provide 2 inputs that are used for training:
        states - These are the pixel values of the game are displayed on the screen
        rewards - These are rewards given by the game to the player. In space invaders, this is the points scored for shooting an enemy
    
Files:

    main.py: Run to trains the RL Agent on space invaders
    player.py: Run to tests the RL Agent on playing space invaders trained on 650 episodes of Space Invaders. Can use this to see the training data for the agent, by setting it's exploration rate to 0
    model.py: Contains the convolutional neural network model and implementation for Deep Q Learning
    sp_inc_dump.csv: dumps training information of the agent in the format : episode_no, score, time_played
    
    data/: contains few example screenshots of SpaceInvaders, Videos of the trained agent and random agent.
    
    