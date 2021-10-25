# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
PARAMETERS = {
    'SimDRLSR': {
        'env_name':             "SimDRLSR",
        'simulation_speed':     2,
        'number_of_agents':     1,
        'action_size':          4,
        'state_size':           8,
        't_steps':              30,
        'frame_height':         320,
        'frame_width':          240,
        'frame_size':           198,
        'port':                 12375,
        'host':                 '127.0.0.1',
        'actions_names':        ['Wait','Look','Wave','Handshake'],
        'actions':              ['1','2','3','4'],        
        'device':               "cpu",
        'stop_scores':          13.0,
        'scores_window_size':   100,
        'train_episodes':       1800,
        'save_images':          False,
        'solved_score':         0.7,




        #'replay_size':          100000,             # replay buffer size
        #'replay_initial':       10000,              # replay buffer initialize
        'replay_size':          10000,             # replay buffer size
        'replay_initial':       1000,              # replay buffer initialize
        'update_interval':      4,                  # network updating every update_interval steps
        'fix_target_updates':   1,                 # fix the target Q for the fix_target_updates

        'hidden_layers':        [16,32,64,256],           # hidden units and layers of Q-network

        'epsilon_start':        1,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.993,              # factor for decreasing epsilon

        'learning_rate':        4e-4,               # learning rate
        'gamma':                0.99,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        'batch_size':           64,                  # minibatch size

        'nstates':              [16,32,64,256],
        'kernels':              [9,5],
        'strides':              [3,1],
        'poolsize':             2,

        'neutral_reward':       0,
        'hs_success_reward':    1,
        'hs_fail_reward':       -0.1,
        'eg_success_reward':    0,
        'eg_fail_reward':       -0.1,
        'ep_fail_reward':       -1
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #