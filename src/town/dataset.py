import torch

class RMDataset(torch.utils.data.Dataset):
    """
    Dataset de PyTorch que encapsula los encodings y las etiquetas.
    Este archivo permite que tanto el preprocesamiento como el entrenamiento
    importen la misma definición de clase.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Devuelve un diccionario con los tensores para un solo item
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)