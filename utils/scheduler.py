
class WeightDecayScheduler:
    def __init__(self, optimizer, wd_start, wd_end, total_steps):
        self.optimizer = optimizer
        self.wd_start = wd_start
        self.wd_end = wd_end
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        # Clip step_num to total_steps to prevent negative decay if training runs over
        current_step = min(self.step_num, self.total_steps)
        
        # Linear interpolation between start and end
        wd = self.wd_end + (self.wd_start - self.wd_end) * (1 - current_step / self.total_steps)

        for group in self.optimizer.param_groups:
            # Skip updating weight decay for parameters that shouldn't have it (like biases or layer norms)
            if group.get("weight_decay", 0.0) != 0.0 or self.step_num == 0:
                group["weight_decay"] = wd

        self.step_num += 1