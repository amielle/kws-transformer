import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import accuracy
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
      
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # Uncomment if testing with extra fc
        # self.fc3 = nn.Linear(hidden_features, hidden_features)
        # self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        # Uncomment if testing with extra fc
        # x = self.act(x)
        # x = self.fc3(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias) 
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer) 
   

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, mlp_ratio=4., qkv_bias=False,  
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio, qkv_bias, 
                                     act_layer, norm_layer) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def init_weights_vit_timm(module: nn.Module):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class KWSTransformer(LightningModule):
    def __init__(self, num_classes=37, lr=0.001, max_epochs=30, depth=12, embed_dim=64,
                 head=4, patch_dim=192, seqlen=16, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Transformer(dim=embed_dim, num_heads=head, num_blocks=depth, mlp_ratio=4.,
                                   qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.embed = torch.nn.Linear(patch_dim, embed_dim)

        self.fc = nn.Linear(seqlen * embed_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.reset_parameters()


    def reset_parameters(self):
        init_weights_vit_timm(self)
    

    def forward(self, x):
        # Linear projection
        x = self.embed(x)
            
        # Encoder
        x = self.encoder(x)
        x = x.flatten(start_dim=1)

        # Classification head
        x = self.fc(x)
        return x
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # this decays the learning rate to 0 after max_epochs using cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y) * 100.
        return {"preds": y_hat, 'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)