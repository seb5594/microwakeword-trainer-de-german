# German microWakeWord trainer

[Deutsche Anleitung](README.de.md) · [Open notebook](notebooks/train_german_wakeword.ipynb)

Train a German microWakeWord model in Google Colab without typing phonetic symbols or storing the training dataset on your computer. This is a text-driven adaptation of [alfiedennen/microwakeword-trainer](https://github.com/alfiedennen/microwakeword-trainer), built on [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word). The notebook generates samples with the official German Piper voices Thorsten, Pavoque, and Ramona.

## Quick start

1. Open `notebooks/train_german_wakeword.ipynb` in [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**, or open the notebook from this repository after it is published.
2. Choose **Runtime → Change runtime type → A100 GPU** and enable **High-RAM** when available. The original trainer uses considerable memory; other runtimes have not been validated here.
3. Enter your wake word in the first form cell. Example: `Ey Sebastian`, `Kosta`, `Koschta`, `Okay Kosta`, `Juii Scheiße`, or another short German word or phrase.
4. Optionally fill `SPOKEN_TEXT` if Piper should read a different spelling, and `SIMILAR_WORDS` with comma-separated phrases that should **not** trigger the model (for example `Kostas, Kosten, Koscha`).
5. Choose **Runtime → Run all**, grant access to your Google Drive, then listen to the three pronunciation previews. Type `JA` only if **each** voice says the intended wake word clearly.
6. Wait for sample generation and training. The `<name>.tflite` model and `<name>.json` manifest are saved in `MyDrive/wakeword_training_de_<name>/`.

Only the model pair is saved to Drive automatically. The generated WAVs, downloaded datasets, and caches reside in the temporary Colab VM. Your local machine does not train or store the datasets. The first run can take substantial time and several gigabytes of Colab storage; it depends on your Colab runtime and dataset download speeds.

## The three inputs

| Input | Required | Meaning |
| --- | --- | --- |
| `WAKE_WORD` | Yes | Visible name in the manifest and basis for output file names. |
| `SPOKEN_TEXT` | No | Alternative text Piper reads. Leave empty to speak `WAKE_WORD`. This does not change the manifest name. |
| `SIMILAR_WORDS` | No | Comma-separated negative examples, such as `Kostas, Kosten`. Do not include the exact wake word. |

Common assistant phrases are added as negative examples automatically. A phrase containing the exact wake word as separate words is excluded from negatives; for example, `Okay Kosta` is **not** a negative for the `Kosta` model. Add meaningful near misses for your particular word when possible. Very short/common words and long phrases can be hard to distinguish reliably. Synthetic speech lasting about three seconds or more causes the notebook to stop before training.

## Pronunciation and voices

Piper receives ordinary German text, including umlauts and `ß`. It uses its own German text conversion. No pronunciation notation is required. The 20,000 positive examples are generated from Thorsten (8,000), Pavoque (6,000), and Ramona (6,000). MLS and Kerstin are excluded. Negative examples rotate through the same three voices. If any voice pronounces your word incorrectly, stop at the preview and adjust `SPOKEN_TEXT` or choose another phrase; do not accept incorrect training audio.

Changing any input automatically creates a separate working directory, so previously generated WAVs cannot silently be reused. Finished model files with the same wake word name in Drive may be overwritten on a later successful run; keep your own copies if you want to compare revisions.

## Using the exported model

Keep the `.json` and `.tflite` files together. For an ESPHome voice device, copy them to a folder such as `/config/esphome/wakewords/ey_sebastian/` and refer to the manifest in the device YAML:

```yaml
micro_wake_word:
  models:
    - model: wakewords/ey_sebastian/ey_sebastian.json
  on_wake_word_detected:
    - voice_assistant.start:
```

Replace `ey_sebastian` with the generated filename. For Linux Voice Assistant / PiCompose, copy both files to the custom wake word location configured by that installation and select the model there. Check the current project's documentation for the path and selection workflow.

Test in your real room with different people, distances, TV/music, and similar phrases. Tune `probability_cutoff` in the exported JSON only after observing false activations or missed detections; synthetic voices alone do not guarantee reliable detection across real microphones and accents.

## Scope and verification

The notebook configuration, generated commands, and code cells are tested locally without GPU training. A complete Colab training run and quality evaluation on real voice recordings have **not** been performed. The pinned Open Home Foundation framework is fetched at commit `4665173cd35f1cff9a61e06fc427f124766c488e`. A separate `piper-sample-generator` 3.2.0 workaround avoids its unconditional legacy `piper_train` import for ONNX inference. The negative feature datasets still come from `kahrendt/microwakeword` on Hugging Face; this does not install the Kahrendt training code. Those datasets and the voice models have their own terms.

## Credits and license

Notebook structure and training flow are derived from [Alfie Dennen's MIT-licensed trainer](https://github.com/alfiedennen/microwakeword-trainer). The training framework is [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word), licensed under Apache-2.0. Piper voices and training datasets have their own licenses. See [LICENSE](LICENSE) for this repository's code and the retained original MIT notice.
