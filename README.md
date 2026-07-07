# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.9 (tested with PyTorch 2.x)
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak-ng`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)


## Pronunciation Stability Improvements

VITS is known to occasionally mispronounce or skip phonemes at inference even
when the input phonemes are correct. The [VITS2 paper](https://arxiv.org/abs/2307.16430)
attributes much of this to the stochastic duration predictor producing
unnatural durations and to alignment errors made early in training. This fork
adds the following mitigations:

**Opt-in training options** (see `configs/ljs_base_stable.json` /
`configs/vctk_base_stable.json`; enabled via the `model` section):
- `use_duration_discriminator`: adversarial training of the duration predictor
  against a VITS2-style duration discriminator, which produces more natural
  durations and clearer pronunciation. The discriminator is a separate network
  (checkpointed as `DUR_*.pth`) and does not change the synthesizer
  architecture, so the resulting generator stays compatible with the original
  inference code.
- `use_noise_scaled_mas` (+ `mas_noise_scale_initial`, `noise_scale_delta`):
  VITS2's noise-scaled Monotonic Alignment Search. Annealed Gaussian noise is
  added to the alignment scores so MAS explores alternative alignments early in
  training instead of committing to its first solution, yielding more accurate
  phoneme-to-frame alignments.
- `use_sdp: false` switches to the deterministic duration predictor, which
  trades rhythm diversity for maximum pronunciation stability.

**Opt-in architecture changes** (see `configs/ljs_vits2.json` /
`configs/vctk_vits2.json`; these change the synthesizer architecture, so they
require training from scratch — checkpoints are NOT interchangeable with the
original architecture):
- `use_transformer_flows` (+ `flow_transformer_n_layers`): replaces the
  WaveNet blocks in the prior normalizing flow with small transformer blocks
  (VITS2), giving the flow long-range context when transforming the prior.
  The coupling projections are zero-initialized, so each coupling layer starts
  as an identity map and training starts from the same dynamics as the
  baseline.
- `use_spk_conditioned_encoder`: conditions the text encoder on the speaker
  embedding (VITS2), which improves pronunciation and speaker similarity in
  multi-speaker models. It has no effect on single-speaker models.

Note that these architecture options come from unofficial reproductions of
VITS2 (no official code was released). If you observe training instability
with `fp16_run: true`, disable mixed precision before drawing conclusions
about the architecture.

**Inference tips for stability** (no retraining needed): lower
`noise_scale_w` (e.g. 0.6 instead of 0.8) to reduce duration randomness, and
lower `noise_scale` (e.g. 0.5) to keep the acoustic latents closer to the
prior mean; both reduce the chance of slurred or distorted phonemes at some
cost in prosody variety.


## PL-BERT for More Natural Prosody

This fork can condition the text encoder on
[PL-BERT](https://github.com/yl4579/PL-BERT) (Phoneme-Level BERT), a BERT
pretrained on phonemized Wikipedia to predict masked phonemes and the
graphemes they came from. Its phoneme-level hidden states carry contextual /
semantic information that plain phoneme embeddings lack, which improves the
naturalness of prosody (this is the same component that StyleTTS 2 uses for
human-level naturalness). The embeddings are added to the phoneme embeddings
inside the text encoder, so they influence both the prior distribution and
the duration predictor.

Setup:

1. Install the extra dependencies (`transformers`, `PyYAML`), included in
   [requirements.txt](requirements.txt).
2. Download the pretrained English PL-BERT checkpoint — the directory must
   contain `config.yml` and a `step_*.t7` checkpoint. The official one
   (trained 1M steps on English Wikipedia) is available from the
   [PL-BERT repo](https://github.com/yl4579/PL-BERT) (Google Drive link in
   its README), and is also shipped in the StyleTTS 2 repo under
   [`Utils/PLBERT`](https://github.com/yl4579/StyleTTS2/tree/main/Utils/PLBERT).
   Place both files in e.g. `./plbert_checkpoints/english/`.
3. Train with a PL-BERT config (see `configs/ljs_plbert.json` /
   `configs/vctk_plbert.json`; based on the VITS2 configs above):

```sh
python train.py -c configs/ljs_plbert.json -m ljs_plbert
```

The relevant `model` options are `use_plbert` (enable the conditioning),
`plbert_dir` (path to the checkpoint directory) and `plbert_dim` (hidden size
of the checkpoint; 768 for the official one). PL-BERT itself stays frozen and
is not stored in the `G_*.pth` checkpoints — only a small projection layer
(`enc_p.bert_proj`) is added to the synthesizer, and it is zero-initialized,
so you can warm-start from a checkpoint trained without PL-BERT
(`utils.load_checkpoint` keeps the zero initialization for the missing
projection weights and training starts from the exact same model).

Notes:
- PL-BERT was trained on IPA phoneme sequences produced with espeak
  (`phonemizer`), which matches the `english_cleaners2` cleaner used by the
  provided configs. Its symbol set is identical to `text/symbols.py`, and the
  token ids are remapped automatically (unknown symbols fall back to
  PL-BERT's convention of `'U'`). With `add_blank: true` the interleaved
  blanks are stripped before PL-BERT and its output is expanded back onto the
  blank-interleaved grid.
- For non-English models, train or download a PL-BERT for your language
  (e.g. the multilingual PL-BERT from the StyleTTS 2 community) and point
  `plbert_dir` at it.
- At inference you must compute the PL-BERT features for the input text and
  pass them to `infer` — see the PL-BERT section of
  [inference.ipynb](inference.ipynb).
