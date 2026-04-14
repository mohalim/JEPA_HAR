
class WeightDecayScheduler:
    def __init__(self, optimizer, wd_start, wd_end, total_steps):
        self.optimizer = optimizer
        self.wd_start = wd_start
        self.wd_end = wd_end
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        wd = self.wd_end + (self.wd_start - self.wd_end) * \
             (1 - self.step_num / self.total_steps)

        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd

        self.step_num += 1