class Config:
    def __init__(self, env_name, seed, max_ep_num=1000):
        self.env_name = env_name
        self.seed_str = "seed=" + str(seed)
        self.output_path = "results/{}-{}/".format(self.env_name, self.seed_str)
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"

        self.layer_size = 64
        self.n_layers = 2

        self.lr = 3e-3
        self.value_update_freq = 1
        self.gamma = 1.0
        self.max_ep_len = 1000
        self.num_batches = 200
        self.batch_size = 2000
        self.max_ep_num = max_ep_num