import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings.models.base_model import BaseModel


class SimpleAutoencoder(BaseModel):
    """Basic AE Model."""

    def __init__(self, d_model, num_inputs, latent_dim=64, **kwargs):
        """ Simple AE architecture for testing
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
    
        self.encoder = nn.Linear(num_inputs, d_model) 
        self.act_fn = F.tanh 
        
        self.decoder= nn.Linear(d_model, num_inputs) 
        
    def forward(self, x, lengths):
        """Apply AE to input x.
            - x: is a three dimensional tensor (batch, stream, channel)
            - lengths: is a one dimensional tensor (batch,) giving the true
              length of each batch element along the stream dimension
        """
        # Encoder:
        x = self.encoder(x) # reduce dimensionality
        x = self.act_fn(x)

        return self.decoder(x)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        # MODEL specific
        parser.add_argument(
            '--d_model', type=int, default=64,
            # tunable=True, options=[64, 128, 256, 512]
        )
        parser.add_argument(
            '--num_inputs', default=94623, type=int
        )
        return parser

