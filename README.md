# German microWakeWord trainer

[Deutsche Anleitung](README.de.md) · [Open notebook](notebooks/train_german_wakeword.ipynb)

Train a German microWakeWord model in Google Colab without typing phonetic symbols or storing the training dataset on your computer. This is a text-driven adaptation of [alfiedennen/microwakeword-trainer](https://github.com/alfiedennen/microwakeword-trainer), built on [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word). The notebook offers six German Piper voice models; you choose which ones to audition and which ones to use for training.

## Quick start

1. Open `notebooks/train_german_wakeword.ipynb` in [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**, or open the notebook from this repository.
2. Choose **Runtime → Change runtime type → A100 GPU** and enable **High-RAM** when available. The original trainer uses considerable memory; other runtimes have not been validated here.
3. Enter your wake word in the first form cell. Example: `Ey Sebastian`, `Kosta`, `Koschta`, `Okay Kosta`, `Juii Scheiße`, or another short German word or phrase. Optionally fill `SPOKEN_TEXT` if Piper should read a different spelling, and `SIMILAR_WORDS` with comma-separated phrases that should **not** trigger the model (for example `Kostas, Kosten, Koscha`).
4. In that first form, tick the voices you want to audition. Thorsten, Pavoque, and Ramona are ticked by default. Run the cells **through the pronunciation previews**, including the setup and Drive access cells, and listen to every preview.
5. In the **second form**, tick only voices that you auditioned and that say your word correctly. Voices newly enabled in the first form must also be explicitly ticked in this second form if you want to train with them. Run this cell and enter `JA` when the listed voices are correct. At least one voice is required.
6. Run the remaining cells for sample generation and training. The `<name>.tflite` model and `<name>.json` manifest are saved in `MyDrive/wakeword_training_de_<name>/`.

Only the model pair is saved to Drive automatically. The generated WAVs, downloaded datasets, and caches reside in the temporary Colab VM. Your local machine does not train or store the datasets. The first run can take substantial time and several gigabytes of Colab storage; it depends on your Colab runtime and dataset download speeds.

## Text inputs

| Input | Required | Meaning |
| --- | --- | --- |
| `WAKE_WORD` | Yes | Visible name in the manifest and basis for output file names. |
| `SPOKEN_TEXT` | No | Alternative text Piper reads. Leave empty to speak `WAKE_WORD`. This does not change the manifest name. |
| `SIMILAR_WORDS` | No | Comma-separated negative examples, such as `Kostas, Kosten`. Do not include the exact wake word. |

Common assistant phrases are added as negative examples automatically. A phrase containing the exact wake word as separate words is excluded from negatives; for example, `Okay Kosta` is **not** a negative for the `Kosta` model. Add meaningful near misses for your particular word when possible. Very short/common words and long phrases can be hard to distinguish reliably. Synthetic speech lasting about three seconds or more causes the notebook to stop before training.

## Pronunciation and voices

Piper receives ordinary German text, including umlauts and `ß`. It uses its own German text conversion. No pronunciation notation is required. Available voices and their positive sample counts when selected:

| Voice checkbox | Piper model | Positive samples | Initially selected for preview/training |
| --- | --- | ---: | --- |
| Thorsten | `de_DE-thorsten-medium` | 8,000 | Yes |
| Pavoque | `de_DE-pavoque-low` | 6,000 | Yes |
| Ramona | `de_DE-ramona-low` | 6,000 | Yes |
| Karlsson | `de_DE-karlsson-low` | 6,000 | No |
| Eva K | `de_DE-eva_k-x_low` | 6,000 | No |
| Thorsten Emotional | `de_DE-thorsten_emotional-medium` | 4,000 | No |

The defaults generate 20,000 positive examples; selecting all six generates 36,000. The `eva_k` model is **x_low**; `eva_k/low` does not refer to this voice. The emotional model exposes eight speaking styles (including whisper and drunk) from Thorsten. Its eight previews let you check all styles; they are not eight independent speakers. If any selected voice or style sounds wrong, untick the entire voice in the second form. You can instead adjust `SPOKEN_TEXT` and rerun the first form and previews. Negative examples rotate through the final voice selection. MLS and Kerstin are excluded.

More distinct, convincing voices can improve coverage, but simply increasing the number of synthetic samples is no guarantee of better detection. Extra voices also add generation time and disk use in Colab. Test the finished model with real speakers to judge whether the extra voices help.

Changing the word, spoken text, negatives, or final voice selection creates a separate working directory, so previously generated WAVs cannot silently be reused. Finished model files with the same wake word name in Drive may be overwritten on a later successful run; keep your own copies if you want to compare revisions.

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
