class Config:
    def __init__(self,
                 env_name,
                 seed,
                 max_ep_num=1000,
                 batch_size=2000,
                 buffer_batch_size=100,
                 max_buffer_size=None,
                 gamma=1.0):
        self.env_name = env_name
        self.seed_str = "seed=" + str(seed)
        self.output_path = "results/{}-{}/".format(self.env_name, self.seed_str)
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"

        self.layer_size = 64
        self.n_layers = 2

        self.lr = 3e-3
        self.value_update_freq = 1
        self.gamma = gamma
        self.max_ep_len = 1000
        self.num_batches = 200
        self.batch_size = batch_size
        self.max_ep_num = max_ep_num

        self.save_freq = 200

        #sac
        self.num_iter = 2000
        self.update_gradient_freq = 25
        self.tau = 0.005
        self.q_lr = 3e-3
        self.v_lr = 3e-3
        self.pi_lr = 3e-3
        self.buffer_batch_size = buffer_batch_size
        self.max_buffer_size = max_buffer_size
        self.alpha = 1.0
        self.explore_step = 200
        
        