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
                if isinstance(node, ast.ImportFrom) and node.module == "utils.utils"
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
            self.assertTrue(
                all(len(call.args) == 4 for call in wrap_calls), relative_path
            )

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
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), "DDPM/train.py", "exec"
            ),
            namespace,
        )
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
            and any(
                isinstance(target, ast.Name) and target.id == "defaults"
                for target in node.targets
            )
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
        in_memory = next(
            keyword.value
            for keyword in loader_call.keywords
            if keyword.arg == "in_memory"
        )
        self.assertEqual("args.in_memory", ast.unparse(in_memory))

    def test_no_data_parallel_device_misuse(self) -> None:
        """Regression: torch.nn.DataParallel(device) вместо модуля."""
        for relative_path in ("VAE/vae.py", "GAN/gan.py", "infer/infer.py"):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            bad_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "DataParallel"
            ]
            self.assertEqual([], bad_calls, relative_path)

    def test_gan_does_not_use_bce_loss(self) -> None:
        """Regression: BCELoss под autocast запрещён PyTorch; нужен BCEWithLogitsLoss."""
        source = (ROOT / "GAN/train.py").read_text(encoding="utf-8")
        self.assertNotIn("BCELoss(", source)
        self.assertIn("BCEWithLogitsLoss()", source)

        gan_tree = ast.parse(source)
        criterion_assigns = [
            node
            for node in ast.walk(gan_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "criterion" for t in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
        ]
        self.assertTrue(criterion_assigns)
        for node in criterion_assigns:
            self.assertEqual("BCEWithLogitsLoss", node.value.func.attr)

    def test_gan_discriminator_has_no_sigmoid(self) -> None:
        """Discriminator выдаёт логиты для BCEWithLogitsLoss — Sigmoid не нужен."""
        tree = ast.parse((ROOT / "GAN/gan.py").read_text(encoding="utf-8"))
        discriminator = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Discriminator"
        )
        sigmoid_calls = [
            node
            for node in ast.walk(discriminator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Sigmoid"
        ]
        self.assertEqual([], sigmoid_calls)

    def test_utils_provides_compare_models(self) -> None:
        """Regression: infer/infer.py импортирует compare_models из utils.utils."""
        source = (ROOT / "utils/utils.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("compare_models", functions)

        infer_source = (ROOT / "infer/infer.py").read_text(encoding="utf-8")
        infer_tree = ast.parse(infer_source)
        imported = {
            alias.name
            for node in ast.walk(infer_tree)
            if isinstance(node, ast.ImportFrom) and node.module == "utils.utils"
            for alias in node.names
        }
        self.assertIn("compare_models", imported)

    def test_gan_train_uses_unwrapped_modules_across_ddp(self) -> None:
        """Regression DDP: в шаге D используется raw_generator, в шаге G — raw_discriminator."""
        tree = ast.parse((ROOT / "GAN/train.py").read_text(encoding="utf-8"))
        train_epoch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "train_epoch"
        )
        calls = [
            node.func.id
            for node in ast.walk(train_epoch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "unwrap_model"
        ]
        self.assertEqual(2, len(calls))

    def test_gan_supports_label_smooth_and_n_critic(self) -> None:
        """Конфиг-параметры label_smooth и n_critic должны читаться кодом."""
        source = (ROOT / "GAN/train.py").read_text(encoding="utf-8")
        self.assertIn('"label_smooth"', source)
        self.assertIn('"n_critic"', source)

    def test_gan_attention_uses_scaled_dot_product(self) -> None:
        """Regression OOM: ручной matmul материализует карту внимания 4096x4096."""
        tree = ast.parse((ROOT / "GAN/gan.py").read_text(encoding="utf-8"))
        attn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "SelfAttention2d"
        )
        calls = {
            node.func.id
            for node in ast.walk(attn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("matmul", calls)
        source = (ROOT / "GAN/gan.py").read_text(encoding="utf-8")
        self.assertIn("scaled_dot_product_attention", source)

    def test_ddpm_sampling_wraps_tqdm_over_list(self) -> None:
        """Regression: tqdm-обёртка не должна подменять tensor timesteps,
        к которому потом обращаются по индексу."""
        source = (ROOT / "DDPM/ddpm.py").read_text(encoding="utf-8")
        # в ddim_sample: список, а не тензор
        self.assertIn("timesteps.tolist()", source)
        self.assertNotIn("prev_t = int(\n                    timesteps[", source)
        # .item() на элементах обёрнутой последовательности недопустим
        self.assertNotRegex(source, r"timesteps\[index \+ 1\]\.item\(\)")
        self.assertNotRegex(source, r"t_idx_tensor\.item\(\)")


if __name__ == "__main__":
    unittest.main()
