from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint

class CallbacksHandler:
    def __init__(self, args, checkpoint_dir):
        self.args = args
        self.checkpoint_dir = checkpoint_dir      
        self.callbacks = self.create_callbacks()
    
    def create_callbacks(self):
        callbacks = []
        
        # Early Stopping Callback
        if getattr(self.args, "enable_early_stopping", True):
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
            
        if getattr(self.args, "enable_model_checkpointing", False):
            model_checkpointing = ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                save_last=True,
                save_top_k=1,
                filename="best-{epoch}-{val_loss:.2f}",
            )
            callbacks.append(model_checkpointing)

        return callbacks