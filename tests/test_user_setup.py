import builtins
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recipe
import setup as setup_entry


class ModelFolderSelection(unittest.TestCase):
    def test_saved_folder_with_spaces_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model_dir = root / "My Models" / "Gyro M"
            model_dir.mkdir(parents=True)
            (model_dir / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf").touch()
            saved = root / ".model-dir"
            saved.write_text(str(model_dir), encoding="utf-8")
            with patch.object(
                builtins, "input", side_effect=AssertionError("unexpected prompt")
            ):
                self.assertEqual(
                    setup_entry.select_model_directory(None, saved), model_dir.resolve()
                )

    def test_explicit_path_overrides_saved_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            saved = root / ".model-dir"
            saved.write_text("old folder", encoding="utf-8")
            chosen = root / "New Folder" / "Gyro"
            with patch.object(
                builtins, "input", side_effect=AssertionError("unexpected prompt")
            ):
                self.assertEqual(
                    setup_entry.select_model_directory(str(chosen), saved),
                    chosen.resolve(),
                )

    def test_stale_saved_folder_prompts_for_new_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            saved = root / ".model-dir"
            saved.write_text(str(root / "moved model"), encoding="utf-8")
            chosen = root / "Current Model Folder"
            chosen.mkdir()
            with patch.object(builtins, "input", return_value=f'"{chosen}"'):
                self.assertEqual(
                    setup_entry.select_model_directory(None, saved), chosen.resolve()
                )


class APIKeyChoice(unittest.TestCase):
    def test_default_install_removes_key_and_opt_in_generates_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = pathlib.Path(tmp) / ".gyro-recipe"
            runtime.mkdir()
            (runtime / "api-key.txt").write_text(
                "old-key-that-should-go", encoding="utf-8"
            )
            config = runtime / "M.json"
            config.write_text(
                json.dumps({"api_key": "old-key-that-should-go"}), encoding="utf-8"
            )
            self.assertEqual(recipe.configure_install_api_key(runtime), "")
            self.assertFalse((runtime / "api-key.txt").exists())
            self.assertNotIn("api_key", json.loads(config.read_text(encoding="utf-8")))
            key = recipe.configure_install_api_key(runtime, api_key=True)
            self.assertGreaterEqual(len(key), 24)

    def test_no_key_removes_keyfile_and_saved_config_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = pathlib.Path(tmp) / ".gyro-recipe"
            backup = runtime / "rollback" / "old"
            backup.mkdir(parents=True)
            key = "test-local-key-that-must-be-removed"
            (runtime / "api-key.txt").write_text(key, encoding="utf-8")
            config_paths = [
                runtime / "M.json",
                runtime / "M.json.previous",
                backup / "M.json",
            ]
            for path in config_paths:
                path.write_text(
                    json.dumps({"api_key": key, "port": 8080}), encoding="utf-8"
                )
            self.assertEqual(recipe.configure_api_key(runtime, True), "")
            self.assertFalse((runtime / "api-key.txt").exists())
            for path in config_paths:
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn("api_key", config)
                self.assertNotIn(key, path.read_text(encoding="utf-8"))

    def test_existing_install_can_be_made_keyless_without_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = pathlib.Path(tmp) / ".gyro-recipe"
            runtime.mkdir()
            key = "test-local-key-that-must-be-removed"
            (runtime / "M.json").write_text(
                json.dumps({"api_key": key, "port": 8080}), encoding="utf-8"
            )
            (runtime / "api-key.txt").write_text(key, encoding="utf-8")
            setup_entry.delete_api_key(tmp)
            self.assertFalse((runtime / "api-key.txt").exists())
            self.assertNotIn(
                "api_key", json.loads((runtime / "M.json").read_text(encoding="utf-8"))
            )

    def test_delete_key_refuses_uninstalled_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No installed Gyro-M profile"):
                setup_entry.delete_api_key(tmp)

    def test_keyed_install_generates_once_and_keeps_existing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = pathlib.Path(tmp) / ".gyro-recipe"
            runtime.mkdir()
            first = recipe.configure_api_key(runtime, False)
            self.assertGreaterEqual(len(first), 24)
            self.assertEqual(recipe.configure_api_key(runtime, False), first)


if __name__ == "__main__":
    unittest.main()
