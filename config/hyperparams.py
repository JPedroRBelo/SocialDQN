# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
PARAMETERS = {
    'SimDRLSR': {
        'env_name':             "simDRLSR",
        'simulation_speed':     1,
        'number_of_agents':     15,
        'action_size':          4,
        'state_size':           8,
        't_steps':              25,

        'frame_height':         264,
        'frame_width':          240,
        'frame_size':           198,
        'port':                 12375,
        'host':                 '127.0.0.1',
        'actions_names':        ['Wait','Look','Wave','Handshake'],
        'actions':              ['1','2','3','4'],        
        'device':               "cpu",
        'stop_scores':          1.0,
        'scores_window_size':   100,
        'train_episodes':       15000,
        'save_images':          False,
        'save_social_states':   True,
        'solved_score':         0.75,
        'stop_when_solved':     False,
        'save_interval':        1000,

        'socket_time_out':      20.0,

        #Multimodal DQN: get detph states
        'use_depth_state':      False,
        'use_only_depth_state': False,
        #WARNING: this mode turns all images to full black, in order to test the input of social signals
        'blind_mode':           False,
        #Social DQN params
        'enable_social_signs':  True,
        #Social State Sizes: 4 = emotional states; 2 = facial_states
        'social_state_size':    4,
        'nstates_social':       [256],
        'neutral_emotions':     ['neutral','unknown'],
        'no_face':              ['no face'],
        'positive_emotions':    ['happy','surprised'],
        'negative_emotions':    ['fear','sad','angry','disgust'],
        'emotional_states':     ['no_face','neutral','positive','negative'],
        'facial_states':        ['no_face','face'],

        #SimDRLSR config
        #Neutral,Anger,Contempt,Enjoyment,Fear,Sadness,Surprise,random
        'emotion_type':         'random',
        'robot_random_position': False,
        'human_appearance':     'default',
        'screen_width':         320,
        'screen_height':        240,
        'quality':              "very low",

        'replay_size':          60000,             # replay buffer size
        #'replay_initial':       10000,              # replay buffer initialize
        #'replay_size':          50000,             # replay buffer size
        'replay_initial':       500,              # replay buffer initialize
        'update_interval':      1,                  # network updating every update_interval steps
        'fix_target_updates':   4,                # fix the target Q for the fix_target_updates
        'train_after_episodes': True,

        #'hidden_layers':        [16,32,64,256],           # hidden units and layers of Q-network



        'epsilon_start':        1.0,                # starting value of epsilon
        #'epsilon_start':        0.01,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.999,              # factor for decreasing epsilon

        'learning_rate':        25e-5,               # learning rate
        'gamma':                0.99,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        #'batch_size':           25,                  # minibatch size
        'batch_size':           64,                  # minibatch size

        'nstates':              [16,32,64,256],
        #'kernels':              [4,2],
        'kernels':              [9,5],
        'strides':              [3,1],
        'poolsize':             2,

        'neutral_reward':       0,
        'hs_success_reward':    1,
        #'hs_fail_reward':       -0.1,
        'hs_fail_reward':       -0.2,
        'eg_success_reward':    0,
        #'eg_fail_reward':    0,
        'eg_fail_reward':       -0.1,
        'ep_fail_reward':       0
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #