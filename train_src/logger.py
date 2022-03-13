import logging
from typing import Optional
from torchinfo import summary
import wandb

class WBLogger(logging.getLoggerClass()):
    def __init__(self, name: str, log_path: str, level, config):
        super().__init__(name, level=level)
        
        msgfmt = '%(name)s: %(asctime)s | %(message)s'
        datfmt = '%H:%M:%S'
        self.formatter = logging.Formatter(fmt=msgfmt, datefmt=datfmt)
        handlers: list[logging.Handler] = [
            logging.StreamHandler(), 
            logging.FileHandler(log_path, 'w', 'utf-8')
        ]
        for h in handlers:
            h.setFormatter(self.formatter)
            self.addHandler(h)
        
        self.run = wandb.init(project='UltrasonicImage', config={})
        self.run.name = name + f"_{wandb.run.id}"
        self.run.save()
        wandb.config.update(config)
        
    def LogMetadata(self, config):
        # save exp parameters as json or yaml...
        
        pass
    
    @staticmethod
    def MergeStringOfDict(log_dict: dict):
        return ", ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])

    def LogTrainingDB(
            self, epoch, max_epoch, datalen, max_datalen, iteration, log_dict: dict, img_dict: Optional[dict] = None):
        self.info(f'[{epoch+1}/{max_epoch}] Train {datalen+1}/{max_datalen} | {self.MergeStringOfDict(log_dict)}')
        
        log_dict.update({
            'Epoch': epoch,
        })
        if img_dict:
            log_dict.update(img_dict)

        wandb.log(log_dict, step=iteration)
    
    def LogValidationDB(self, epoch, max_epoch, iteration, log_dict: dict, img_dict: Optional[dict] = None):
        self.info(f'[{epoch+1}/{max_epoch}] Valid | {self.MergeStringOfDict(log_dict)}')
        log_dict = { f'Valid_{k}': v for k, v in log_dict.items() }
        log_dict.update({
            'Epoch': epoch,
        })
        if img_dict:
            for k, v in img_dict.items():
                log_dict[f'Valid_{k}'] = wandb.Image(v) 

        wandb.log(log_dict, step=iteration)

    def LogSummary(self, log_dict: dict):
        for k, v in log_dict.items():
            self.run.summary[k] = v
    
    def LogNet(self, net):
        s = summary(net, verbose=0)
        wandb.watch(net)
        wandb.config['NetParams'] = s.total_params
        wandb.config['NetSummary'] = str(s)