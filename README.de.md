# Deutscher microWakeWord-Trainer

[English guide](README.md) · [Notebook öffnen](notebooks/train_german_wakeword.ipynb)

Mit diesem Google-Colab-Notebook trainierst du ein deutsches microWakeWord-Modell, ohne Lautschrift einzugeben und ohne die Trainingsdaten auf deinem PC zu speichern. Es basiert auf [alfiedennen/microwakeword-trainer](https://github.com/alfiedennen/microwakeword-trainer) und verwendet [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word). Für Hörproben und Training stehen sechs deutsche Piper-Stimmmodelle zur Auswahl.

## Schnellstart

1. Öffne `notebooks/train_german_wakeword.ipynb` in [Google Colab](https://colab.research.google.com/) über **Datei → Notebook hochladen** oder direkt aus diesem GitHub-Repository.
2. Wähle unter **Laufzeit → Laufzeittyp ändern** eine **A100 GPU** und nach Möglichkeit **hohen RAM**. Der zugrunde liegende Trainer benötigt viel Speicher; andere Laufzeiten wurden hier nicht vollständig geprüft.
3. Trage in der ersten Formularzelle dein `WAKE_WORD` ein, zum Beispiel `Ey Sebastian`, `Kosta`, `Koschta`, `Okay Kosta` oder ein anderes kurzes deutsches Wort. Optional: `SPOKEN_TEXT` ist eine andere Schreibweise nur für die Sprachausgabe. In `SIMILAR_WORDS` kannst du mit Kommas getrennte Wörter eintragen, die das Modell **nicht** aktivieren sollen, etwa `Kostas, Kosten, Koscha`.
4. Markiere im **ersten Formular** die Stimmen, von denen du Hörproben haben möchtest. Thorsten, Pavoque und Ramona sind voreingestellt. Führe die Zellen **bis einschließlich der Hörproben** aus und verbinde dabei Google Drive. Höre dir alle Proben an.
5. Markiere im **zweiten Formular** nur die zuvor angehörten Stimmen, die dein Wort richtig aussprechen. Neu ausgewählte Stimmen müssen dort nochmals ausdrücklich angehakt werden. Führe diese Zelle aus und bestätige die angezeigten Stimmen mit `JA`. Mindestens eine Stimme muss ausgewählt bleiben.
6. Führe anschließend die restlichen Zellen für Datenerzeugung und Training aus. Danach findest du `<name>.tflite` und `<name>.json` in `MyDrive/wakeword_training_german/<name>/`.

Das Notebook speichert automatisch nur die beiden fertigen Modelldateien in Drive. WAV-Dateien, Zwischenergebnisse und große Hintergrund-Datensätze liegen in der temporären Colab-VM. Dein PC muss sie nicht speichern und trainiert nichts. Ein Durchlauf kann lange dauern und mehrere Gigabyte Speicher in Colab benötigen.

## Texteingaben

| Feld | Pflicht | Zweck |
| --- | --- | --- |
| `WAKE_WORD` | Ja | Anzeigename im Modell und Grundlage des Dateinamens. |
| `SPOKEN_TEXT` | Nein | Alternative Schreibweise für Piper; leer bedeutet `WAKE_WORD`. Der Anzeigename bleibt gleich. |
| `SIMILAR_WORDS` | Nein | Ähnlich klingende Wörter als Gegenbeispiele, durch Kommas getrennt. |

Andere übliche Assistenten-Rufwörter werden automatisch als Gegenbeispiele ergänzt. Phrasen, in denen das genaue Wakeword als eigenständige Wortfolge vorkommt, werden ausgeschlossen: `Okay Kosta` ist beispielsweise kein Gegenbeispiel für `Kosta`. Trage nach Möglichkeit gezielt ähnlich klingende Wörter ein. Sehr kurze oder alltägliche Wörter und lange Phrasen lassen sich schwerer zuverlässig erkennen. Dauern die synthetischen Aufnahmen ungefähr drei Sekunden oder länger, stoppt das Notebook vor dem Training.

## Aussprache und Stimmen

Piper erhält normalen deutschen Text, auch mit Umlauten und `ß`. Für jede ausgewählte Stimme wird folgende Anzahl positiver Beispiele erzeugt:

| Stimme | Piper-Modell | Positive Beispiele | Anfangs für Vorschau/Training markiert |
| --- | --- | ---: | --- |
| Thorsten | `de_DE-thorsten-medium` | 8.000 | Ja |
| Pavoque | `de_DE-pavoque-low` | 6.000 | Ja |
| Ramona | `de_DE-ramona-low` | 6.000 | Ja |
| Karlsson | `de_DE-karlsson-low` | 6.000 | Nein |
| Eva K | `de_DE-eva_k-x_low` | 6.000 | Nein |
| Thorsten Emotional | `de_DE-thorsten_emotional-medium` | 4.000 | Nein |

Voreingestellt sind 20.000 positive Beispiele; mit allen sechs Modellen sind es 36.000. Bei Eva K heißt die Qualitätsstufe **x_low**; `eva_k/low` ist dafür kein gültiger Pfad. Das emotionale Thorsten-Modell hat acht Sprechstile, darunter Flüstern und Betrunkensein. Du kannst alle acht anhören; es sind keine acht unabhängigen Sprecher. Falls eine Stimme oder einer ihrer Stile falsch klingt, entferne die gesamte Stimme im zweiten Formular. Alternativ kannst du `SPOKEN_TEXT` ändern und das erste Formular samt Vorschauen erneut ausführen. Die Gegenbeispiele verwenden abwechselnd die endgültig gewählten Stimmen. MLS und Kerstin sind nicht enthalten.

Mehr unterschiedliche und überzeugende Stimmen können helfen. Mehr künstliche Beispiele allein garantieren aber keine bessere Erkennung. Jede zusätzliche Stimme benötigt Zeit und Speicherplatz in Colab. Ob die Auswahl wirklich besser ist, lässt sich erst mit echten Sprechern prüfen.

Bei Änderungen am Wort, gesprochenen Text, den Gegenbeispielen oder der endgültigen Stimmauswahl legt das Notebook automatisch einen neuen Arbeitsordner an. So werden keine WAV-Dateien eines anderen Wortes oder einer alten Aussprache übernommen. Ein erneuter erfolgreicher Durchlauf mit demselben Wakeword kann die Modelldateien gleichen Namens in Drive überschreiben. Sichere Versionen zum Vergleichen separat.

## Modell verwenden

Die `.json`- und `.tflite`-Datei gehören zusammen. In ESPHome kannst du sie beispielsweise nach `/config/esphome/wakewords/ey_sebastian/` kopieren und in der Gerätekonfiguration referenzieren:

```yaml
micro_wake_word:
  models:
    - model: wakewords/ey_sebastian/ey_sebastian.json
  on_wake_word_detected:
    - voice_assistant.start:
```

Ersetze `ey_sebastian` durch den erzeugten Dateinamen. Bei Linux Voice Assistant / PiCompose legst du beide Dateien am dort konfigurierten Ort für eigene Wakewords ab und wählst das Modell in dessen Oberfläche aus. Der genaue Ort hängt von der verwendeten Version ab.

Teste mit verschiedenen Personen, Abständen und Hintergrundgeräuschen in deinem Raum. Bei Fehlauslösungen oder ausbleibender Erkennung kannst du nach Messung die `probability_cutoff` in der JSON-Datei anpassen. Synthetische Stimmen allein garantieren keine zuverlässige Erkennung echter Stimmen und Mikrofone.

## Prüfstand

Die Notebook-Zellen, Eingaben und erzeugten Befehle wurden lokal geprüft. Ein vollständiger GPU-Trainingslauf in Colab und ein Qualitätsvergleich mit echten Sprachaufnahmen stehen noch aus. Das Trainingsframework stammt von der Open Home Foundation und ist auf Commit `4665173cd35f1cff9a61e06fc427f124766c488e` festgelegt. Ein separater Workaround verhindert den überflüssigen `piper_train`-Import in `piper-sample-generator` 3.2.0. Die negativen Merkmals-Datensätze stammen weiterhin von `kahrendt/microwakeword` auf Hugging Face; der Trainingscode wird dadurch nicht von Kahrendt installiert. Für diese Datensätze und die Stimmen gelten eigene Lizenzbedingungen.

## Herkunft und Lizenz

Das Notebook ist eine Bearbeitung des [MIT-lizenzierten Trainers von Alfie Dennen](https://github.com/alfiedennen/microwakeword-trainer). Das Trainingsframework [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word) steht unter Apache-2.0. Die ursprüngliche MIT-Nennung bleibt in [LICENSE](LICENSE) erhalten.
