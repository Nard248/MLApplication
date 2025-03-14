import torch
from torch.utils.data import Dataset, DataLoader


class GLDataset(Dataset):
    def __init__(self, state, myu):
        if isinstance(state, torch.Tensor):
            self.states = state.clone().detach().to(dtype=torch.complex64)
            self.myus = myu.clone().detach().to(dtype=torch.float32)
        else:
            self.states = torch.tensor(state, dtype=torch.complex64)
            self.myus = torch.tensor(myu, dtype=torch.float32)

        if self.states.shape != self.myus.shape:
            raise ValueError(f"Shape mismatch: states shape {self.states.shape} \
                and myus shape {self.myus.shape} must match in all dimensions.")

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        state = self.states[idx]
        myus = self.myus[idx]
        return state, myus


class GLDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=False):
        if not isinstance(dataset, GLDataset):
            print("Dataset type is not GLDataset")
            return
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

class StateDataset(torch.utils.data.Dataset):
    def __init__(self, state, myu):
        self.inputs = []
        self.outputs = []

        t_dim, x_dim, y_dim = state.shape

        for t in range(t_dim):
            for x in range(x_dim):
                for y in range(y_dim):
                    self.inputs.append([t / t_dim, x / x_dim, y / y_dim])
                    self.outputs.append([state[t, x, y].real, state[t, x, y].imag])

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]