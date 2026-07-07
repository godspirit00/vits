import torch
from torch import nn

import commons
from text.symbols import symbols as vits_symbols
from .util import load_plbert

# PL-BERT phoneme vocabulary, verbatim from
# https://github.com/yl4579/PL-BERT/blob/main/text_utils.py. It is identical
# to this repo's text/symbols.py except for the padding glyph ('$' vs '_'),
# because both were derived from the same IPA symbol set.
_pad = '$'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_plbert_symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)


def _build_token_map():
  """LongTensor mapping VITS symbol ids to PL-BERT token ids."""
  plbert_ids = {s: i for i, s in enumerate(_plbert_symbols)}
  unknown = plbert_ids['U']  # PL-BERT's TextCleaner maps unknown symbols to 'U'
  token_map = [0]  # VITS pad/blank -> PL-BERT pad
  for s in vits_symbols[1:]:
    token_map.append(plbert_ids.get(s, unknown))
  return torch.LongTensor(token_map)


class PLBertEmbedder(nn.Module):
  """Frozen PL-BERT that turns VITS text-token sequences into phoneme-level
  contextual embeddings aligned with the input tokens.

  Takes the same padded token ids the synthesizer consumes (including the
  interspersed blanks when `add_blank` is enabled), strips the blanks, runs
  the pretrained PL-BERT over the phoneme sequence, and expands the hidden
  states back onto the original token grid (each blank receives the embedding
  of the adjacent phoneme). Output shape: [batch, hidden_size, text_len].
  """
  def __init__(self, plbert_dir, add_blank=True):
    super().__init__()
    self.bert = load_plbert(plbert_dir)
    self.hidden_size = self.bert.config.hidden_size
    self.add_blank = add_blank
    self.register_buffer('token_map', _build_token_map(), persistent=False)
    self.bert.eval()
    for p in self.bert.parameters():
      p.requires_grad = False

  def train(self, mode=True):
    super().train(mode)
    self.bert.eval()  # stays frozen
    return self

  @torch.no_grad()
  def forward(self, x, x_lengths):
    """
    x: [b, t] padded VITS token ids
    x_lengths: [b]
    returns: [b, hidden_size, t] float32
    """
    if self.add_blank:
      # interspersed sequence is [blank, p0, blank, p1, ..., blank]
      ph = x[:, 1::2]
      ph_lengths = torch.clamp(x_lengths // 2, min=1)
    else:
      ph, ph_lengths = x, x_lengths
    ids = self.token_map[ph]
    attn_mask = commons.sequence_mask(ph_lengths, ph.size(1)).long()
    hidden = self.bert(ids, attention_mask=attn_mask)  # [b, t_ph, h]
    if self.add_blank:
      # position p of the interspersed sequence maps to phoneme p // 2
      # (each blank takes the embedding of the phoneme that follows it,
      # the trailing blank that of the last phoneme)
      idx = torch.arange(x.size(1), device=x.device) // 2
      idx = torch.clamp(idx, max=hidden.size(1) - 1)
      hidden = hidden[:, idx]
    return hidden.transpose(1, 2).float()
