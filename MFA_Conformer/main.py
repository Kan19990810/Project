from pytorch_lightning import LightningModule
class Task(LightningModule):
    def __init__(self, learning_rate, weight_decay, batch_size, num_workers, max_epochs, trial_path, **kwargs):
        super().__init__()
        self