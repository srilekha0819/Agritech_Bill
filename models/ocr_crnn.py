import torch
import torch.nn as nn
from typing import List, Tuple

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for multilingual OCR.
    Architecture:
      1. 7-layer Deep CNN Feature Extractor
      2. Bidirectional LSTM Sequence Modeler
      3. Linear Projection to Character Classes
      4. CTC Loss compatibility
    """
    def __init__(
        self,
        img_h: int = 32,
        in_channels: int = 1,
        num_classes: int = 150,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super(CRNN, self).__init__()
        self.img_h = img_h
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Convolutional layers
        # Input: (B, 1, H, W) — H can be 32, 48, or any config.OCR_IMG_H
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, H/2, W/2)

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, H/4, W/4)

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Block 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 256, H/8, W/4)

            # Conv Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Block 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 512, H/16, W/4)

            # Adaptive height collapse — works for any OCR_IMG_H
            # Squeezes remaining height dimension to 1, keeps full width
            nn.AdaptiveAvgPool2d((1, None)),  # (B, 512, 1, W/4)
        )

        # Bidirectional LSTM for sequence modeling
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Linear classifier projecting to vocabulary size
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # Suppress blank token initial bias so CTC gradients flow into character classes
        with torch.no_grad():
            self.fc.bias.data[0] = -2.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Height, Width)
               Height can be any value (32, 48, 64...) — AdaptiveAvgPool handles it.
        Returns:
            Logits of shape (SeqLen, Batch, NumClasses) for CTC Loss
        """
        # Feature extraction
        conv_feats = self.cnn(x)  # (B, 512, 1, W')
        b, c, h, w = conv_feats.size()
        # Height should be 1 from AdaptiveAvgPool2d — flatten it
        conv_feats = conv_feats.view(b, c, -1)  # (B, 512, W') — handles h>=1 safely

        # Squeeze height: (B, C, W') -> permute to (W', B, C) for RNN
        rnn_input = conv_feats.permute(2, 0, 1)  # (W', B, 512)

        # RNN Sequence modeling
        rnn_out, _ = self.rnn(rnn_input)  # (W', B, 2 * hidden_size)

        # Output projection
        logits = self.fc(rnn_out)  # (W', B, num_classes)
        return logits

    def decode_greedy(self, logits: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
        """
        CTC Greedy Decoding:
        Finds argmax class per timestep, compresses repeated identical tokens,
        and eliminates blank tokens.
        
        Args:
            logits: Tensor of shape (T, Batch, NumClasses)
            blank_idx: Index of CTC blank token
        Returns:
            List of decoded token IDs for each sample in batch.
        """
        probs = torch.softmax(logits, dim=2)
        max_indices = torch.argmax(probs, dim=2)  # (T, Batch)
        max_indices = max_indices.permute(1, 0).cpu().numpy()  # (Batch, T)

        decoded_batch = []
        for seq in max_indices:
            decoded_seq = []
            prev = None
            for token in seq:
                if token != prev:
                    if token != blank_idx:
                        decoded_seq.append(int(token))
                    prev = token
            decoded_batch.append(decoded_seq)
        return decoded_batch

    def decode_beam_search(
        self,
        logits: torch.Tensor,
        beam_width: int = 10,
        blank_idx: int = 0
    ) -> List[List[int]]:
        """
        CTC Prefix Beam Search:
        Accumulates probabilities over distinct alignments producing the same prefix.
        Eliminates the premature blank collapse problem of greedy decoding.
        
        Args:
            logits: Tensor of shape (T, Batch, NumClasses)
            beam_width: Number of active candidate prefixes to maintain per timestep
            blank_idx: Index of CTC blank token
        Returns:
            List of decoded token IDs for each sample in batch.
        """
        import numpy as np
        probs = torch.softmax(logits, dim=2).detach().cpu().numpy()  # (T, Batch, C)
        seq_len, batch_size, num_classes = probs.shape

        decoded_batch = []
        for b in range(batch_size):
            # beams maps prefix tuple to (p_blank, p_non_blank)
            beams = {(): (1.0, 0.0)}

            for t in range(seq_len):
                p_t = probs[t, b]  # (C,)
                # Select top candidates to evaluate
                top_k = min(beam_width * 2, num_classes)
                top_indices = np.argpartition(p_t, -top_k)[-top_k:]

                next_beams = {}
                for prefix, (p_b, p_nb) in beams.items():
                    p_total = p_b + p_nb
                    if p_total <= 1e-9:
                        continue

                    for c in top_indices:
                        pr_c = float(p_t[c])
                        if pr_c <= 1e-6:
                            continue

                        if c == blank_idx:
                            cur_b, cur_nb = next_beams.get(prefix, (0.0, 0.0))
                            next_beams[prefix] = (cur_b + p_total * pr_c, cur_nb)
                        else:
                            last_token = prefix[-1] if prefix else None
                            if c == last_token:
                                # Repeated token without blank -> stays same prefix
                                cur_b, cur_nb = next_beams.get(prefix, (0.0, 0.0))
                                next_beams[prefix] = (cur_b, cur_nb + p_nb * pr_c)

                                # Repeated token with blank -> extends prefix
                                new_prefix = prefix + (int(c),)
                                cur_b2, cur_nb2 = next_beams.get(new_prefix, (0.0, 0.0))
                                next_beams[new_prefix] = (cur_b2, cur_nb2 + p_b * pr_c)
                            else:
                                new_prefix = prefix + (int(c),)
                                cur_b, cur_nb = next_beams.get(new_prefix, (0.0, 0.0))
                                next_beams[new_prefix] = (cur_b, cur_nb + p_total * pr_c)

                # Prune to top beam_width
                if next_beams:
                    sorted_beams = sorted(
                        next_beams.items(),
                        key=lambda item: item[1][0] + item[1][1],
                        reverse=True
                    )
                    beams = dict(sorted_beams[:beam_width])
                else:
                    break

            if beams:
                best_prefix = max(beams.items(), key=lambda item: item[1][0] + item[1][1])[0]
                decoded_batch.append(list(best_prefix))
            else:
                decoded_batch.append([])

        return decoded_batch

