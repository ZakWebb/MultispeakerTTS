import pytorch_lightning
from pytorch_lightning.callbacks import Callback

class ProfCallback(Callback):
    def on_exception(self, trainer, pl_module, exception):
        trainer.profiler.describe() 