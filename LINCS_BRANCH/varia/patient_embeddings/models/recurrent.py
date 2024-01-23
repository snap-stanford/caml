import torch
import torch.nn as nn
from embeddings.models.base_model import BaseModel


class RecurrentAutoencoder(BaseModel):
    """Recurrent Model."""

    def __init__(self, model_name, d_model, n_layers, dropout, num_inputs, latent_dim=64, **kwargs):
        """Recurrent Model.
        Args:
            -model_name: ['GRU', 'LSTM', 'RNN']
            -d_model: Dimensionality of hidden state of the recurrent layers
            -n_layers: Number of stacked RNN layers
            -dropout: Fraction of elements that should be dropped out,
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        model_cls = getattr(torch.nn, model_name)
    

        self.first_linear = nn.Linear(num_inputs, d_model) 
        self.enc_model = model_cls(
            hidden_size=d_model,
            input_size=d_model,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True) # expects (batch, length, feature)
        self.latent_linear1 = nn.Linear(d_model, latent_dim)
        self.latent_linear2 = nn.Linear(latent_dim, d_model)
        self.dec_model = model_cls(
            hidden_size=d_model,
            input_size=d_model,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True) # expects (batch, length, feature)
        self.last_linear = nn.Linear(d_model, num_inputs) 
        

    def forward(self, x, lengths):
        """Apply GRU model to input x.
            - x: is a three dimensional tensor (batch, stream, channel)
            - lengths: is a one dimensional tensor (batch,) giving the true
              length of each batch element along the stream dimension
        """
        # Encoder:
        x = self.first_linear(x) # reduce dimensionality

        ## Convert padded sequence to packed sequence for increased efficiency
        #x = nn.utils.rnn.pack_padded_sequence(
        #    x, lengths, batch_first=True, enforce_sorted=False)
        hidden_init = None
        
        x, hidden_states = self.enc_model(x, hidden_init)
        x = self.latent_linear1(x)

        #x = torch.nn.utils.rnn.PackedSequence(
        #    self.linear(x.data),
        #    x.batch_sizes,
        #    x.sorted_indices,
        #    x.unsorted_indices
        #)
        # Unpack sequence to padded tensor for subsequent operations
        #x, lenghts = torch.nn.utils.rnn.pad_packed_sequence(
        #    x, batch_first=True)

        # Decoder: 
        x = self.latent_linear2(x)
        x, hidden_states = self.dec_model(x, hidden_init) #TODO: use better init?
        return self.last_linear(x)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        # MODEL specific
        parser.add_argument(
            '--d_model', type=int, default=64,
            # tunable=True, options=[64, 128, 256, 512]
        )
        parser.add_argument(
            '--n_layers', type=int, default=2,
            # tunable=True, options=[1, 2, 3]
        )
        parser.add_argument(
            '--dropout', default=0.1, type=float,
            # tunable=True, options=[0., 0.1, 0.2, 0.3, 0.4]
        )
        parser.add_argument(
            '--num_inputs', default=94623, type=int
        )
        return parser


class GRUAutoencoder(RecurrentAutoencoder):
    """
    GRU Model.
    """
    def __init__(self, **kwargs):
        model_name = 'GRU'
        if 'model_name' in kwargs.keys():
            del kwargs['model_name']
        super().__init__(model_name, **kwargs)
        self.save_hyperparameters()


class LSTMAutoencoder(RecurrentAutoencoder):
    """
    LSTM Model.
    """
    def __init__(self, **kwargs):
        model_name = 'LSTM'
        if 'model_name' in kwargs.keys():
            del kwargs['model_name']
        super().__init__(model_name, **kwargs)
        self.save_hyperparameters()


class RNNAutoencoder(RecurrentAutoencoder):
    """
    RNN Model.
    """
    def __init__(self, **kwargs):
        model_name = 'RNN'
        if 'model_name' in kwargs.keys():
            del kwargs['model_name']
        super().__init__(model_name, **kwargs)
        self.save_hyperparameters()
