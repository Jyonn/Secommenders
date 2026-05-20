from torch.utils.data import DataLoader

from .dataset import CompiledSampleDataset


def build_dataloaders(compiled, batch_size: int):
    finetune = CompiledSampleDataset(compiled.finetune)
    test = CompiledSampleDataset(compiled.test)
    train_rows = [finetune[index] for index in range(len(finetune))]
    test_rows = [test[index] for index in range(len(test))]

    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)
    test_loader = DataLoader(test_rows, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch)
    return train_loader, test_loader
