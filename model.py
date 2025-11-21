import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)  # outputs: (batch, seq_len, hidden)

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        score = torch.bmm(self.attn(encoder_outputs), decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, hidden_size=64, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(output_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.attention = LuongAttention(hidden_size)

    def forward(self, y_prev, hidden, encoder_outputs):
        # y_prev: (batch, 1, 1)
        h = hidden[0][-1]  # (batch, hidden)
        context, attn = self.attention(h, encoder_outputs)
        lstm_input = torch.cat([y_prev.squeeze(1), context.unsqueeze(1)], dim=2)
        out, hidden = self.lstm(lstm_input, hidden)
        pred = self.fc(out.squeeze(1))
        return pred.unsqueeze(1), hidden, attn

class Seq2SeqAttention(nn.Module):
    def __init__(self, enc_params={}, dec_params={}, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(**enc_params)
        self.decoder = Decoder(**dec_params)

    def forward(self, src, target_len):
        # src: (batch, src_len, 1)
        encoder_outputs, hidden = self.encoder(src)
        batch = src.size(0)
        y_prev = src[:, -1:, :]  # last value as start token
        outputs = []
        for t in range(target_len):
            y_pred, hidden, attn = self.decoder(y_prev, hidden, encoder_outputs)
            outputs.append(y_pred)
            y_prev = y_pred
        return torch.cat(outputs, dim=1)  # (batch, target_len, 1)
