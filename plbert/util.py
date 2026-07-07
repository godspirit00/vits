""" PL-BERT (https://github.com/yl4579/PL-BERT, MIT license) loading
utilities, adapted from the loader shipped with StyleTTS2
(https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py). """
import os

import torch
import yaml
from transformers import AlbertConfig, AlbertModel


class CustomAlbert(AlbertModel):
  """AlbertModel that returns only the last hidden state."""
  def forward(self, *args, **kwargs):
    outputs = super().forward(*args, **kwargs)
    return outputs.last_hidden_state


def load_plbert(log_dir):
  """Load a pretrained PL-BERT from its checkpoint directory.

  The directory must contain the `config.yml` the model was trained with and
  at least one `step_*.t7` (or `.pt`/`.pth`) checkpoint; the latest step is
  loaded. This matches the layout of the official pretrained checkpoint.
  """
  config_path = os.path.join(log_dir, "config.yml")
  with open(config_path) as f:
    plbert_config = yaml.safe_load(f)

  albert_config = AlbertConfig(**plbert_config['model_params'])
  bert = CustomAlbert(albert_config)

  ckpts = []
  for fname in os.listdir(log_dir):
    root, ext = os.path.splitext(fname)
    if root.startswith("step_") and ext in (".t7", ".pt", ".pth"):
      ckpts.append((int(root.split("_")[-1]), os.path.join(log_dir, fname)))
  if not ckpts:
    raise FileNotFoundError(
        "No PL-BERT checkpoint (step_*.t7) found in {}".format(log_dir))
  ckpt_path = max(ckpts)[1]

  try:
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
  except Exception:
    # older checkpoints may contain non-tensor pickled objects
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  state_dict = checkpoint['net']
  new_state_dict = {}
  for k, v in state_dict.items():
    if k.startswith('module.'):
      k = k[len('module.'):]
    # drop the masked-phoneme/word prediction heads, keep the encoder
    if k.startswith('encoder.'):
      new_state_dict[k[len('encoder.'):]] = v
  # transformers >= 4.31 keeps position_ids as a non-persistent buffer
  new_state_dict.pop("embeddings.position_ids", None)
  bert.load_state_dict(new_state_dict, strict=False)
  return bert
