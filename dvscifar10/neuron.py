from spikingjelly.clock_driven.neuron import ParametricLIFNode, LIFNode
from typing import Callable
from spikingjelly.clock_driven import surrogate
import torch


class ParametricLIAFNode(ParametricLIFNode):
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(init_tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        output = torch.nn.ReLU()(self.v)
        self.neuronal_reset(spike)
        return output


class MultiStepParametricLIAFNode(ParametricLIAFNode):
    def __init__(self, init_tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        super().__init__(init_tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        assert backend == 'torch'
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                self.v_seq.append(super().forward(x_seq[t]).unsqueeze(0))

            self.v_seq = torch.cat(self.v_seq, 0)

            return self.v_seq

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'


class LIAFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        output = torch.nn.ReLU()(self.v)
        self.neuronal_reset(spike)
        return output


class MultiStepLIAFNode(LIAFNode):
    def __init__(self, init_tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        super().__init__(init_tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        assert backend == 'torch'
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                self.v_seq.append(super().forward(x_seq[t]).unsqueeze(0))

            self.v_seq = torch.cat(self.v_seq, 0)

            return self.v_seq

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'
