import torch


class NormalizationAdapter(torch.nn.Module):
    def __init__(self, mean_original, std_original, mean_required, std_required):
        super(NormalizationAdapter, self).__init__()
        mean_original = torch.tensor(mean_original).unsqueeze(-1).unsqueeze(-1)
        std_original = torch.tensor(std_original).unsqueeze(-1).unsqueeze(-1)
        mean_required = torch.tensor(mean_required).unsqueeze(-1).unsqueeze(-1)
        std_required = torch.tensor(std_required).unsqueeze(-1).unsqueeze(-1)

        self.additive = torch.nn.Parameter((mean_original - mean_required) / std_original)
        self.multiplier = torch.nn.Parameter(std_original / std_required)

    def forward(self, x):
        x = (x + self.additive) * self.multiplier
        return x
