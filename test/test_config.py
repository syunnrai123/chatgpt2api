import json
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_CONFIG_FILE = ROOT_DIR / "config.json"


class ConfigLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._created_root_config = False
        if not ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            cls._created_root_config = True

        from services import config as config_module

        cls.config_module = config_module

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created_root_config and ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.unlink()

    def test_load_settings_ignores_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            data_dir = base_dir / "data"
            config_dir = base_dir / "config.json"
            os_auth_key = "env-auth"

            config_dir.mkdir()

            module = self.config_module
            old_base_dir = module.BASE_DIR
            old_data_dir = module.DATA_DIR
            old_config_file = module.CONFIG_FILE
            old_env_auth_key = module.os.environ.get("CHATGPT2API_AUTH_KEY")
            try:
                module.BASE_DIR = base_dir
                module.DATA_DIR = data_dir
                module.CONFIG_FILE = config_dir
                module.os.environ["CHATGPT2API_AUTH_KEY"] = os_auth_key

                settings = module._load_settings()

                self.assertEqual(settings.auth_key, os_auth_key)
                self.assertEqual(settings.refresh_account_interval_minute, 5)
            finally:
                module.BASE_DIR = old_base_dir
                module.DATA_DIR = old_data_dir
                module.CONFIG_FILE = old_config_file
                if old_env_auth_key is None:
                    module.os.environ.pop("CHATGPT2API_AUTH_KEY", None)
                else:
                    module.os.environ["CHATGPT2API_AUTH_KEY"] = old_env_auth_key

    def test_image_retention_minutes_and_save_flag_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")

            store = self.config_module.ConfigStore(path)
            data = store.update({
                "image_save_enabled": False,
                "image_retention_minutes": "45",
            })

            self.assertFalse(data["image_save_enabled"])
            self.assertEqual(data["image_retention_minutes"], 45)
            self.assertEqual(data["image_retention_days"], 1)

    def test_image_save_defaults_to_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")

            store = self.config_module.ConfigStore(path)

            self.assertFalse(store.get()["image_save_enabled"])

    def test_legacy_image_retention_days_populates_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")

            store = self.config_module.ConfigStore(path)
            data = store.update({"image_retention_days": "2"})

            self.assertEqual(data["image_retention_minutes"], 2880)

    def test_proxy_and_volatile_image_limits_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")

            store = self.config_module.ConfigStore(path)
            data = store.update({
                "trust_proxy_headers": "true",
                "trusted_proxy_ips": "127.0.0.1 10.0.0.0/8",
                "max_volatile_image_results": "2",
                "max_volatile_image_bytes": "1024",
            })

            self.assertTrue(data["trust_proxy_headers"])
            self.assertEqual(data["trusted_proxy_ips"], ["127.0.0.1", "10.0.0.0/8"])
            self.assertEqual(data["max_volatile_image_results"], 2)
            self.assertEqual(data["max_volatile_image_bytes"], 1024)

    def test_invalid_trusted_proxy_ip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")

            store = self.config_module.ConfigStore(path)

            with self.assertRaisesRegex(ValueError, "IP 规则无效"):
                store.update({"trusted_proxy_ips": ["not-an-ip"]})

    def test_legacy_invalid_trusted_proxy_ip_does_not_block_unrelated_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps({
                    "auth-key": "test-auth",
                    "trusted_proxy_ips": ["not-an-ip"],
                    "base_url": "",
                }),
                encoding="utf-8",
            )

            store = self.config_module.ConfigStore(path)
            data = store.update({"base_url": "https://example.test"})

            self.assertEqual(data["base_url"], "https://example.test")
            self.assertEqual(data["trusted_proxy_ips"], [])

    def test_trust_proxy_headers_requires_trusted_proxy_ips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")

            store = self.config_module.ConfigStore(path)

            with self.assertRaisesRegex(ValueError, "可信反代 IP"):
                store.update({"trust_proxy_headers": True, "trusted_proxy_ips": []})

            with self.assertRaisesRegex(ValueError, "可信反代 IP"):
                store.update({"trust_proxy_headers": True, "trusted_proxy_ips": ""})


if __name__ == "__main__":
    unittest.main()
