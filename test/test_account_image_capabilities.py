from __future__ import annotations

import os
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend
from utils.helper import anonymize_token


class AccountCapabilityTests(unittest.TestCase):
    def test_unknown_quota_accounts_are_available_only_when_not_throttled(self) -> None:
        self.assertFalse(
            AccountService._is_image_account_available(
                {"status": "限流", "image_quota_unknown": True, "quota": 0}
            )
        )
        self.assertTrue(
            AccountService._is_image_account_available(
                {"status": "正常", "image_quota_unknown": True, "quota": 0}
            )
        )

    def test_2k_and_4k_require_non_free_image_account(self) -> None:
        self.assertTrue(AccountService._supports_image_resolution({"type": "free"}, "1k"))
        self.assertFalse(AccountService._supports_image_resolution({"type": "free"}, "2k"))
        self.assertFalse(AccountService._supports_image_resolution({"type": "free"}, "4k"))
        self.assertTrue(AccountService._supports_image_resolution({"type": "Plus"}, "2k"))
        self.assertTrue(AccountService._supports_image_resolution({"type": "Pro"}, "4k"))

    def test_get_available_access_token_filters_free_accounts_for_2k_and_4k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([
                {"access_token": "free-token", "type": "free", "quota": 1, "status": "正常"},
                {"access_token": "plus-token", "type": "Plus", "quota": 1, "status": "正常"},
            ])
            service.fetch_remote_info = lambda token, _event: service.get_account(token)

            token = service.get_available_access_token("4k")

            self.assertEqual(token, "plus-token")
            service.release_image_slot(token)

    def test_get_available_access_token_rejects_2k_when_only_free_accounts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([
                {"access_token": "free-token", "type": "free", "quota": 1, "status": "正常"},
            ])
            service.fetch_remote_info = lambda token, _event: service.get_account(token)

            with self.assertRaisesRegex(RuntimeError, "non-free account"):
                service.get_available_access_token("2k")

    def test_prolite_variants_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertEqual(service._normalize_account_type("prolite"), "ProLite")
            self.assertEqual(service._normalize_account_type("pro_lite"), "ProLite")

    def test_search_account_type_ignores_unrelated_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertIsNone(
                service._search_account_type(
                    {
                        "amr": ["pwd", "otp", "mfa"],
                        "chatgpt_compute_residency": "no_constraint",
                        "chatgpt_data_residency": "no_constraint",
                        "user_id": "user-I52GFfLGFM0dokFk2dBiKEBn",
                    }
                )
            )

    def test_mark_image_result_does_not_consume_unknown_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_accounts(["token-1"])
            service.update_account(
                "token-1",
                {
                    "status": "正常",
                    "quota": 0,
                    "image_quota_unknown": True,
                },
            )

            updated = service.mark_image_result("token-1", success=True)

            self.assertIsNotNone(updated)
            self.assertEqual(updated["quota"], 0)
            self.assertEqual(updated["status"], "正常")
            self.assertTrue(updated["image_quota_unknown"])


class TokenLogTests(unittest.TestCase):
    def test_anonymize_token_hides_raw_value(self) -> None:
        token = "super-secret-token"
        token_ref = anonymize_token(token)

        self.assertTrue(token_ref.startswith("token:"))
        self.assertNotIn(token, token_ref)


class AuthServiceTests(unittest.TestCase):
    def test_create_authenticate_disable_and_delete_user_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))

            item, raw_key = service.create_key(role="user", name="Alice")

            self.assertEqual(item["role"], "user")
            self.assertEqual(item["name"], "Alice")
            self.assertTrue(item["enabled"])
            self.assertTrue(raw_key.startswith("sk-"))

            authed = service.authenticate(raw_key)
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertEqual(authed["role"], "user")
            self.assertIsNotNone(authed["last_used_at"])

            updated = service.update_key(item["id"], {"enabled": False}, role="user")
            self.assertIsNotNone(updated)
            self.assertFalse(updated["enabled"])
            self.assertIsNone(service.authenticate(raw_key))

            self.assertTrue(service.delete_key(item["id"], role="user"))
            self.assertFalse(service.delete_key(item["id"], role="user"))
            self.assertEqual(service.list_keys(role="user"), [])

    def test_authenticate_ignores_last_used_save_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            def fail_save() -> None:
                raise OSError("disk unavailable")

            service._save = fail_save

            authed = service.authenticate(raw_key)

            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertIsNotNone(authed["last_used_at"])

    def test_update_user_key_replaces_raw_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            updated = service.update_key(item["id"], {"key": "sk-user-custom-key"}, role="user")

            self.assertIsNotNone(updated)
            self.assertIsNone(service.authenticate(raw_key))

            authed = service.authenticate("sk-user-custom-key")
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])

    def test_user_key_name_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            first, _ = service.create_key(role="user", name="Alice")
            second, _ = service.create_key(role="user", name="Bob")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.create_key(role="user", name="Alice")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.update_key(second["id"], {"name": "Alice"}, role="user")

            updated = service.update_key(first["id"], {"name": "Alice"}, role="user")
            self.assertIsNotNone(updated)
            self.assertEqual(updated["name"], "Alice")

    def test_legacy_user_key_defaults_to_zero_image_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth_keys_path = Path(tmp_dir) / "auth_keys.json"
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", auth_keys_path))
            item, _ = service.create_key(role="user", name="Alice", image_quota=5)
            payload = json.loads(auth_keys_path.read_text(encoding="utf-8"))
            payload["items"][0].pop("image_quota", None)
            payload["items"][0].pop("image_quota_reserved", None)
            auth_keys_path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", auth_keys_path))
            [listed] = reloaded.list_keys(role="user")

            self.assertEqual(listed["id"], item["id"])
            self.assertEqual(listed["image_quota"], 0)
            self.assertEqual(listed["image_quota_available"], 0)

    def test_image_quota_reserve_confirm_and_refund(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            service._image_quota_multiplier = lambda _mode: Decimal("1")
            item, raw_key = service.create_key(role="user", name="Alice", image_quota=2.5)
            identity = service.authenticate(raw_key)

            reservation = service.reserve_image_quota(identity, mode="generate", count=2)
            duplicate = service.reserve_image_quota(identity, mode="generate", count=2, reservation_id=reservation["id"])
            self.assertEqual(duplicate["id"], reservation["id"])
            listed = service.list_keys(role="user")[0]
            self.assertEqual(listed["image_quota"], 2.5)
            self.assertEqual(listed["image_quota_reserved"], 2)
            self.assertEqual(listed["image_quota_available"], 0.5)

            service.refund_image_quota(reservation)
            listed = service.list_keys(role="user")[0]
            self.assertEqual(listed["image_quota"], 2.5)
            self.assertEqual(listed["image_quota_available"], 2.5)

            reservation = service.reserve_image_quota(identity, mode="edit", count=1)
            service.confirm_image_quota(reservation)
            service.confirm_image_quota(reservation)
            service.refund_image_quota(reservation)
            listed = service.list_keys(role="user")[0]
            self.assertEqual(listed["image_quota"], 1.5)
            self.assertEqual(listed["image_quota_reserved"], 0)

            with self.assertRaisesRegex(ValueError, "图片额度不足"):
                service.reserve_image_quota(identity, mode="generate", count=2)

    def test_expired_sync_image_quota_reservation_is_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth_keys_path = Path(tmp_dir) / "auth_keys.json"
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", auth_keys_path))
            service._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = service.create_key(role="user", name="Alice", image_quota=1)
            identity = service.authenticate(raw_key)
            reservation = service.reserve_image_quota(identity, mode="generate", count=1)
            payload = json.loads(auth_keys_path.read_text(encoding="utf-8"))
            stored_reservation = payload["items"][0]["image_quota_reservations"][reservation["id"]]
            stored_reservation["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            auth_keys_path.write_text(json.dumps(payload), encoding="utf-8")

            [listed] = service.list_keys(role="user")
            payload = json.loads(auth_keys_path.read_text(encoding="utf-8"))

            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)
            self.assertEqual(listed["image_quota_available"], 1)
            self.assertEqual(payload["items"][0]["image_quota_reservations"], {})


if __name__ == "__main__":
    unittest.main()
