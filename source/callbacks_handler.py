from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging, LearningRateMonitor

class CallbacksHandler:
    def __init__(self, args):
        self.args = args
        self.callbacks = self.create_callbacks()
    
    def create_callbacks(self):
        callbacks = []
        
        # Early Stopping Callback
        if getattr(self.args, "enable_early_stopping", False):
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=getattr(self.args, "early_stopping_patience", 10),
                mode="min"
            )
            callbacks.append(early_stop)
        
        # Learning Rate Monitor Callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # SWA Callback
        if getattr(self.args, "enable_swa", False):
            swa = StochasticWeightAveraging(
                swa_lrs=getattr(self.args, "swa_lrs", 0.01),
                swa_epoch_start=getattr(self.args, "swa_epoch_start", 0.8),
                annealing_epochs=getattr(self.args, "annealing_epochs", 10),
                annealing_strategy=getattr(self.args, "annealing_strategy", "cos")
            )
            callbacks.append(swa)

        return callbacks