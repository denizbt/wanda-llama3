import torch


class WrappedGPT2:
    """Collect Wanda input statistics for a GPT-2 Conv1D projection."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        # Conv1D stores [input, output], while Wanda uses [output, input].
        self.columns = layer.weight.shape[0]
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = inp.float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

