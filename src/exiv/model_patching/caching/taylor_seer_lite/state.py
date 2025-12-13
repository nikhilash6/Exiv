import math
import torch
from typing import List, Dict

# same as the main taylor seer, will cleanup things later
class TaylorSeerState:
    def __init__(self, n_derivatives=1, max_warmup_steps=3, skip_interval_steps=1):
        self.n_derivatives = n_derivatives
        self.order = n_derivatives + 1
        self.max_warmup_steps = max_warmup_steps
        # computes how often the step is calculated, if skip_interval_steps = 2
        # then it calculates 1 step and approximates 1 step (check the tests)
        self.skip_interval_steps = skip_interval_steps
        self.reset()

    def reset(self):
        self.state: Dict[str, List[torch.Tensor]] = {
            "dY_prev": [None] * self.order,
            "dY_current": [None] * self.order,
        }
        self.current_step = -1
        self.last_non_approximated_step = -1

    def mark_step_begin(self):
        self.current_step += 1

    def should_compute(self):
        if (self.current_step < self.max_warmup_steps or 
            (self.current_step - self.max_warmup_steps + 1) % self.skip_interval_steps == 0):
            return True
        return False

    def derivative(self, Y: torch.Tensor) -> List[torch.Tensor]:
        dY_current = [None] * self.order
        dY_current[0] = Y
        window = self.current_step - self.last_non_approximated_step
        
        if self.state["dY_prev"][0] is not None:
             if dY_current[0].shape != self.state["dY_prev"][0].shape:
                self.reset()
                dY_current = [None] * self.order
                dY_current[0] = Y
                return dY_current

        for i in range(self.n_derivatives):
            if self.state["dY_prev"][i] is not None and self.current_step > 0:
                dY_current[i + 1] = (dY_current[i] - self.state["dY_prev"][i]) / window
            else:
                break
        return dY_current

    def approximate(self) -> torch.Tensor:
        elapsed = self.current_step - self.last_non_approximated_step
        output = 0
        for i, derivative in enumerate(self.state["dY_current"]):
            if derivative is not None:
                output += (1 / math.factorial(i)) * derivative * (elapsed**i)
            else:
                break
        return output

    def update(self, Y: torch.Tensor):
        self.state["dY_prev"] = self.state["dY_current"]
        self.state["dY_current"] = self.derivative(Y)
        self.last_non_approximated_step = self.current_step