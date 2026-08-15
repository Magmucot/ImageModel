from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingScriptTests(unittest.TestCase):
    def test_model_imports_point_to_existing_modules(self) -> None:
        expected = {
            "VAE/train.py": "VAE.vae",
            "GAN/train.py": "GAN.gan",
        }

        for relative_path, expected_module in expected.items():
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            self.assertIn(expected_module, imported_modules, relative_path)
            self.assertTrue(
                (ROOT / Path(*expected_module.split("."))).with_suffix(".py").is_file(),
                expected_module,
            )

    def test_vae_and_gan_use_nested_config_reader(self) -> None:
        for relative_path in ("VAE/train.py", "GAN/train.py"):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            imported_names = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "utils.utils"
                for alias in node.names
            }
            self.assertIn("get_config_value", imported_names, relative_path)

            direct_cfg_gets = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cfg"
                and node.func.attr == "get"
            ]
            self.assertEqual([], direct_cfg_gets, relative_path)

    def test_vae_and_gan_match_distributed_helper_api(self) -> None:
        for relative_path in ("VAE/train.py", "GAN/train.py"):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            setup_assignment = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "setup_distributed"
            )
            target = setup_assignment.targets[0]
            self.assertIsInstance(target, ast.Tuple, relative_path)
            self.assertEqual(4, len(target.elts), relative_path)

            wrap_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "wrap_ddp"
            ]
            self.assertTrue(wrap_calls, relative_path)
            self.assertTrue(all(len(call.args) == 4 for call in wrap_calls), relative_path)

    def test_vae_and_gan_do_not_force_per_process_memory_cache(self) -> None:
        for relative_path in ("VAE/train.py", "GAN/train.py"):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            loader_call = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_dataloaders"
            )
            in_memory = next(
                keyword.value
                for keyword in loader_call.keywords
                if keyword.arg == "in_memory"
            )
            self.assertIsInstance(in_memory, ast.Call, relative_path)
            self.assertIsInstance(in_memory.func, ast.Name, relative_path)
            self.assertEqual("get_config_value", in_memory.func.id, relative_path)
            self.assertEqual("in_memory", in_memory.args[1].value, relative_path)
            self.assertIs(in_memory.args[2].value, False, relative_path)

    def test_ddpm_dry_run_skips_sampling(self) -> None:
        source = (ROOT / "DDPM/train.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "should_sample"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "DDPM/train.py", "exec"), namespace)
        should_sample = namespace["should_sample"]

        self.assertFalse(should_sample(2, 2, 2, True))
        self.assertTrue(should_sample(2, 10, 2, False))
        self.assertTrue(should_sample(10, 10, 20, False))
        self.assertFalse(should_sample(1, 10, 2, False))

    def test_ddpm_memory_cache_is_configurable_and_off_by_default(self) -> None:
        tree = ast.parse((ROOT / "DDPM/train.py").read_text(encoding="utf-8"))

        defaults = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "defaults" for target in node.targets)
            and isinstance(node.value, ast.Dict)
        )
        default_values = {
            key.value: value.value
            for key, value in zip(defaults.keys, defaults.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        }
        self.assertIs(default_values["in_memory"], False)

        loader_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_dataloaders"
        )
        in_memory = next(keyword.value for keyword in loader_call.keywords if keyword.arg == "in_memory")
        self.assertEqual("args.in_memory", ast.unparse(in_memory))


if __name__ == "__main__":
    unittest.main()
