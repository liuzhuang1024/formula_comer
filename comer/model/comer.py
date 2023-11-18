from typing import List

# import pytorch_lightning as pl
from torch import nn
import torch
from torch import FloatTensor, LongTensor

# from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class CoMER(nn.Module):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        **kwargs
    ):
        super().__init__()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
                
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            **kwargs
        )
            
    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d] # mask: 1 is mask position
        if hasattr(self, 'bttr') and self.bttr:
            # close BTTR hand 
            feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
            mask = torch.cat((mask, mask), dim=0)
        out = self.decoder(feature, mask, tgt)
        return out
    
    def inference(self, img, img_mask, max_len, **kwargs):
        if not hasattr(self, 'device'):
            setattr(self, 'device', img.device)

        feature, mask = self.encoder(img, img_mask)  # [b, t, d] # mask: 1 is mask position
        if hasattr(self, 'bttr') and self.bttr:
            # close BTTR hand 
            feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
            mask = torch.cat((mask, mask), dim=0)
        
        # attention tgt shape
        batch = feature.shape[0]
        preds = torch.ones(batch, 1, dtype=torch.long).to(self.device) 
        for index in range(1, max_len+1):
            out = self.decoder(feature, mask, preds[:, :index])
            preds = torch.cat([preds, torch.argmax(out, dim=-1)], dim=-1)
        return out
    
    def stop(self, out):
        return False
    
    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ):
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
