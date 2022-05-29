import torch
import time
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

from utils.dataloader import KWSDataModule
from utils.model import KWSTransformer

def get_args(parser):
    # model training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    # where dataset will be stored
    parser.add_argument("--path", type=str, default="data/speech_commands/")

    # 35 keywords + silence + unknown
    parser.add_argument("--num-classes", type=int, default=37)
   
    # mel spectrogram parameters
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)

    # 16-bit fp model to reduce the size
    parser.add_argument("--precision", default=16)
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--num-workers", type=int, default=2)

    # transformer arguments
    parser.add_argument('--depth', type=int, default=32, help='depth')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='num_heads')

    parser.add_argument('--patch_num', type=int, default=4, help='patch_num')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')
    
    parser.add_argument("--no-wandb", default=True, action="store_true")

    args = parser.parse_args("")
    return args


class WandbCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # log 10 sample audio predictions from the first batch
        if batch_idx == 0:
            n = 10
            mels, labels, wavs = batch
            preds = outputs["preds"]
            preds = torch.argmax(preds, dim=1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            
            wavs = torch.squeeze(wavs, dim=1)
            wavs = [ (wav.cpu().numpy()*32768.0).astype("int16") for wav in wavs]
            
            sample_rate = pl_module.hparams.sample_rate
            idx_to_class = pl_module.hparams.idx_to_class
            
            # log audio samples and predictions as a W&B Table
            columns = ['audio', 'mel', 'ground truth', 'prediction']
            data = [[wandb.Audio(wav, sample_rate=sample_rate), wandb.Image(mel), idx_to_class[label], idx_to_class[pred]] for wav, mel, label, pred in list(
                zip(wavs[:n], mels[:n], labels[:n], preds[:n]))]
            wandb_logger.log_table(
                key='KWS using Transformer and PyTorch Lightning',
                columns=columns,
                data=data)

if __name__ == "__main__":
    parser = ArgumentParser()
    args = get_args(parser)
    print("No wandb:",args.no_wandb)

    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
            'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    datamodule = KWSDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                path=args.path, n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length,
                                patch_num=args.patch_num,
                                class_dict=CLASS_TO_IDX)
    datamodule.setup()

    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    print("Embed dim:", args.embed_dim)
    print("Sequence length:", seqlen)

    model = KWSTransformer(num_classes=37, lr=args.lr, epochs=args.max_epochs, 
                        depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                        patch_dim=patch_dim, seqlen=seqlen,)


    # wandb is a great way to debug and visualize this model

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.path, "checkpoints"),
        filename="transformer-kws-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    if not args.no_wandb:
        import time
        wandb_logger = WandbLogger(project=f"kws-{time.time()}")
        callbacks = [model_checkpoint, WandbCallback()]
    else:
        wandb_logger = None
        callbacks = [model_checkpoint]

    trainer = Trainer(accelerator=args.accelerator,
                    devices=args.devices,
                    precision=args.precision,
                    max_epochs=args.max_epochs,
                    logger=wandb_logger if not args.no_wandb else None,
                    callbacks=callbacks)
    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    if not args.no_wandb: wandb.finish()

    script = model.to_torchscript()
    model_path = f"{os.getcwd()}/models/transformer-kws-{int(time.time())}.pt"
    torch.jit.save(script, model_path)
