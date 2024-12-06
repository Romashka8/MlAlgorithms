import numpy as np


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

    output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

    gradInput = module.backward(input, gradOutput)
    """
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input_data):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.update_output(input_data)

    def backward(self, input_data, grad_output):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
        - computing a gradient w.r.t. `input` (is needed for further backprop),
        - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.update_grad_input(input_data, grad_output)
        self.acc_grad_parameters(input_data, grad_output)
        return self.gradInput

    def update_output(self, input_data):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """

        # The easiest case:

        # self.output = input
        # return self.output
        pass

    def update_grad_input(self, input_data, grad_output):
        """
                Computing the gradient of the module with respect to its own input.
                This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

                The shape of `gradInput` is always the same as the shape of `input`.

                Make sure to both store the gradients in `gradInput` field and return it.
                """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def acc_grad_parameters(self, input_data, grad_output):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zero_grad_parameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def get_grad_parameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"
