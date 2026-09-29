"""Local smoke tests for the Colab notebook without downloading models."""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'train_german_wakeword.ipynb'


def source(index):
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    return ''.join(cells[index]['source'])


def select_voices(scope, changes=None):
    code = source(6)
    for name, enabled in (changes or {}).items():
        original = f'KEEP_{name.upper()} = False  # @param'
        if original not in code:
            original = f'KEEP_{name.upper()} = True  # @param'
        assert original in code
        code = code.replace(original, f'KEEP_{name.upper()} = {enabled}  # @param')
    scope['os'] = os
    scope['input'] = lambda prompt: 'JA'
    with patch('os.makedirs'), patch('os.chdir'):
        exec(code, scope)


class NotebookTests(unittest.TestCase):
    def test_all_code_cells_compile(self):
        cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
        for index, cell in enumerate(cells):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell_{index}', 'exec')

    def test_examples_slug_negatives_and_isolation(self):
        settings = source(1)
        for word, expected in (
            ('Ey Sebastian', 'ey_sebastian'), ('Kosta', 'kosta'), ('Koschta', 'koschta'),
            ('Okay Kosta', 'okay_kosta'), ('Juii Scheiße', 'juii_scheisse'),
            ('Ey Kosta', 'ey_kosta'), ('Grüß Gott', 'gruess_gott'),
        ):
            lines = settings.replace('WAKE_WORD = "Ey Sebastian"  # @param',
                                     f'WAKE_WORD = {word!r}  # @param')
            scope = {}
            exec(lines, scope)
            self.assertEqual(scope['OUTPUT_NAME'], expected)
            self.assertEqual(scope['DRIVE_FOLDER'], f'wakeword_training_german/{expected}')
            self.assertEqual(scope['SPOKEN_TEXT'], word)
            self.assertFalse(any(scope['contains_target'](phrase)
                                 for phrase in scope['CONFUSABLE_PHRASES']))
            select_voices(scope)
            self.assertEqual(sum(scope['VOICE_SAMPLE_COUNTS'].values()), 20000)
            self.assertEqual(set(scope['VOICE_SPECS']), set(scope['VOICE_SAMPLE_COUNTS']))
            self.assertIn(expected, scope['WORK_DIR'])

        adjusted = settings.replace('WAKE_WORD = "Ey Sebastian"  # @param',
                                    'WAKE_WORD = "Kosta"  # @param')
        adjusted = adjusted.replace('SPOKEN_TEXT = ""  # @param',
                                    'SPOKEN_TEXT = "Koschta"  # @param')
        custom = adjusted.replace('SIMILAR_WORDS = ""  # @param',
                                  'SIMILAR_WORDS = "Kostas, Okay Kosta, Kosten"  # @param')
        scope = {}
        exec(custom, scope)
        select_voices(scope)
        self.assertEqual(scope['SPOKEN_TEXT'], 'Koschta')
        self.assertIn('Kostas', scope['CONFUSABLE_PHRASES'])
        self.assertIn('Kosten', scope['CONFUSABLE_PHRASES'])
        self.assertNotIn('Okay Kosta', scope['CONFUSABLE_PHRASES'])

        default = {}
        exec(settings, default)
        select_voices(default)
        self.assertNotEqual(default['WORK_DIR'], scope['WORK_DIR'])

    def test_optional_voices_and_post_preview_selection(self):
        code = source(1)
        for name in ('KARLSSON', 'EVA_K', 'THORSTEN_EMOTIONAL'):
            code = code.replace(f'PREVIEW_{name} = False  # @param',
                                f'PREVIEW_{name} = True  # @param')
        scope = {}
        exec(code, scope)
        self.assertEqual(len(scope['VOICE_SPECS']), 6)
        self.assertEqual(scope['VOICE_SPECS']['eva_k'],
                         'eva_k/x_low/de_DE-eva_k-x_low')
        select_voices(scope, {'PAVOQUE': 'False', 'KARLSSON': 'True',
                              'EVA_K': 'True', 'THORSTEN_EMOTIONAL': 'True'})
        self.assertEqual(set(scope['VOICE_SAMPLE_COUNTS']),
                         {'thorsten', 'ramona', 'karlsson', 'eva_k', 'thorsten_emotional'})
        self.assertEqual(sum(scope['VOICE_SAMPLE_COUNTS'].values()), 30000)

        not_previewed = {}
        exec(source(1), not_previewed)
        with self.assertRaisesRegex(AssertionError, 'Preview these voices first'):
            select_voices(not_previewed, {'EVA_K': 'True'})

    def test_generation_uses_text_and_approved_voices(self):
        settings = {}
        exec(source(1), settings)
        select_voices(settings)
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            settings['WORK_DIR'] = str(work_dir)
            settings['VOICE_SAMPLE_COUNTS'] = {voice: 2 for voice in settings['VOICE_SPECS']}
            settings['SAMPLES_PER_CONFUSABLE'] = 2
            settings['VOICE_PATHS'] = {voice: str(work_dir / f'{voice}.onnx')
                                       for voice in settings['VOICE_SPECS']}

            class FakeProcess:
                commands = []

                @classmethod
                def run(cls, cmd, check):
                    assert check
                    assert '--phoneme-input' not in cmd
                    cls.commands.append(cmd)
                    output = Path(cmd[cmd.index('--output-dir') + 1])
                    output.mkdir(parents=True, exist_ok=True)
                    for i in range(int(cmd[cmd.index('--max-samples') + 1])):
                        (output / f'{i}.wav').write_bytes(b'RIFF')

            with patch('subprocess.run', FakeProcess.run):
                exec(source(7), settings)
                positive_commands = FakeProcess.commands[:]
                self.assertEqual(len(positive_commands), 3)
                self.assertTrue(all(cmd[3] == 'Ey Sebastian' for cmd in positive_commands))
                self.assertEqual({Path(cmd[cmd.index('--model') + 1]).stem
                                  for cmd in positive_commands},
                                 {'thorsten', 'pavoque', 'ramona'})

                exec(source(8), settings)
                negative_commands = FakeProcess.commands[len(positive_commands):]
                self.assertEqual(len(negative_commands), len(settings['CONFUSABLE_PHRASES']))
                self.assertEqual({Path(cmd[cmd.index('--model') + 1]).stem
                                  for cmd in negative_commands},
                                 {'thorsten', 'pavoque', 'ramona'})

    def test_export_keeps_only_model_pair_in_drive(self):
        scope = {}
        exec(source(1), scope)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained = (root / 'trained_models' / 'ey_sebastian' /
                       'tflite_stream_state_internal_quant')
            trained.mkdir(parents=True)
            (trained / 'stream_state_internal_quant.tflite').write_bytes(b'model')
            drive_dir = root / 'drive' / scope['DRIVE_FOLDER']
            drive_dir.mkdir(parents=True)
            (drive_dir / '_run_finished.txt').write_text('old run', encoding='utf-8')
            scope['DRIVE_DIR'] = str(drive_dir)
            previous_dir = os.getcwd()
            try:
                os.chdir(root)
                exec(source(14), scope)
            finally:
                os.chdir(previous_dir)
            self.assertEqual({file.name for file in drive_dir.iterdir()},
                             {'ey_sebastian.tflite', 'ey_sebastian.json'})
            manifest = json.loads((drive_dir / 'ey_sebastian.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['website'],
                             'https://github.com/seb5594/microwakeword-trainer-de-german')

    def test_no_active_pronunciation_notation(self):
        cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
        active = '\n'.join(line for cell in cells if cell['cell_type'] == 'code'
                           for line in ''.join(cell['source']).splitlines()
                           if not line.lstrip().startswith('#'))
        self.assertNotRegex(active, re.compile(r'\bUSE_IPA_INPUT\b|\bWAKE_WORD_IPA\b'))


if __name__ == '__main__':
    unittest.main()
