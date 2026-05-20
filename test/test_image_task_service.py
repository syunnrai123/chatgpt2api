from __future__ import annotations

import json
import threading
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

import services.image_task_service as image_task_module
import services.image_quota as image_quota_module
from services.image_task_service import ImageTaskService
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend


OWNER = {"id": "owner-1", "name": "Owner", "role": "admin"}
OTHER_OWNER = {"id": "owner-2", "name": "Other", "role": "user"}


def wait_for_task(service: ImageTaskService, identity: dict[str, object], task_id: str, status: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        result = service.list_tasks(identity, [task_id])
        last = (result.get("items") or [None])[0]
        if last and last.get("status") == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {status}, last={last}")


class ImageTaskServiceTests(unittest.TestCase):
    def make_service(
        self,
        path: Path,
        handler=None,
        task_workers: int | None = None,
        task_queue_size: int | None = None,
    ) -> ImageTaskService:
        return ImageTaskService(
            path,
            generation_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/image.png"}]}),
            edit_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/edit.png"}]}),
            retention_days_getter=lambda: 30,
            task_workers=task_workers,
            task_queue_size=task_queue_size,
        )

    def test_task_worker_queue_limits_concurrent_handlers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            started: list[str] = []
            lock = threading.Lock()
            first_started = threading.Event()
            release = threading.Event()

            def handler(payload):
                with lock:
                    started.append(str(payload.get("prompt")))
                    first_started.set()
                release.wait(1)
                return {"data": [{"url": f"http://example.test/{payload.get('prompt')}.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler, task_workers=1)
            service.submit_generation(
                OWNER,
                client_task_id="queued-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            service.submit_generation(
                OWNER,
                client_task_id="queued-2",
                prompt="dog",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            self.assertTrue(first_started.wait(1))
            time.sleep(0.05)
            self.assertEqual(started, ["cat"])

            release.set()
            wait_for_task(service, OWNER, "queued-1", "success")
            wait_for_task(service, OWNER, "queued-2", "success")

    def test_stream_quota_confirms_after_natural_completion(self):
        reservation = {"id": "stream-reservation"}

        def handler(_payload):
            yield {"data": [{"url": "http://example.test/image.png"}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            result = image_quota_module.run_image_handler_with_quota(handler, {}, reservation)
            self.assertEqual(list(result), [{"data": [{"url": "http://example.test/image.png"}]}])

        auth.confirm_image_quota.assert_called_once_with(reservation)
        auth.refund_image_quota.assert_not_called()

    def test_stream_quota_confirms_when_stream_is_closed_after_result(self):
        reservation = {"id": "stream-reservation"}

        def handler(_payload):
            yield {"data": [{"url": "http://example.test/image.png"}]}
            yield {"data": [{"url": "http://example.test/image-2.png"}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            result = image_quota_module.run_image_handler_with_quota(handler, {}, reservation)
            self.assertEqual(next(result), {"data": [{"url": "http://example.test/image.png"}]})
            result.close()

        auth.confirm_image_quota.assert_called_once_with(reservation)
        auth.refund_image_quota.assert_not_called()

    def test_non_stream_quota_confirm_failure_returns_result_and_refunds(self):
        reservation = {"id": "result-reservation"}

        def handler(_payload):
            return {"data": [{"url": "http://example.test/image.png"}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            auth.confirm_image_quota.side_effect = OSError("disk full")
            result = image_quota_module.run_image_handler_with_quota(handler, {}, reservation)

        self.assertEqual(result, {"data": [{"url": "http://example.test/image.png"}]})
        auth.confirm_image_quota.assert_called_once_with(reservation)
        auth.refund_image_quota.assert_called_once_with(reservation)

    def test_non_stream_quota_uses_custom_success_predicate(self):
        reservation = {"id": "chat-image-reservation"}

        def handler(_payload):
            return {"choices": [{"message": {"content": "![image](data:image/png;base64,abc)"}}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            result = image_quota_module.run_image_handler_with_quota(
                handler,
                {},
                reservation,
                success_predicate=lambda item: "data:image/" in str(item),
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "![image](data:image/png;base64,abc)")
        auth.confirm_image_quota.assert_called_once_with(reservation)
        auth.refund_image_quota.assert_not_called()

    def test_stream_quota_confirm_failure_does_not_break_result_stream(self):
        reservation = {"id": "stream-reservation"}

        def handler(_payload):
            yield {"data": [{"url": "http://example.test/image.png"}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            auth.confirm_image_quota.side_effect = OSError("disk full")
            result = image_quota_module.run_image_handler_with_quota(handler, {}, reservation)
            self.assertEqual(list(result), [{"data": [{"url": "http://example.test/image.png"}]}])

        auth.confirm_image_quota.assert_called_once_with(reservation)
        auth.refund_image_quota.assert_called_once_with(reservation)

    def test_stream_quota_retries_refund_when_confirm_and_initial_refund_fail(self):
        reservation = {"id": "stream-reservation"}

        def handler(_payload):
            yield {"data": [{"url": "http://example.test/image.png"}]}
            yield {"data": [{"url": "http://example.test/image-2.png"}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            auth.confirm_image_quota.side_effect = OSError("disk full")
            auth.refund_image_quota.side_effect = [OSError("disk full"), None]
            result = image_quota_module.run_image_handler_with_quota(handler, {}, reservation)
            self.assertEqual(next(result), {"data": [{"url": "http://example.test/image.png"}]})
            result.close()

        auth.confirm_image_quota.assert_called_once_with(reservation)
        self.assertEqual(auth.refund_image_quota.call_count, 2)

    def test_stream_quota_refunds_when_stream_is_closed_before_result(self):
        reservation = {"id": "stream-reservation"}

        def handler(_payload):
            yield {"data": []}
            yield {"data": [{"url": "http://example.test/image.png"}]}

        with mock.patch.object(image_quota_module, "auth_service") as auth:
            result = image_quota_module.run_image_handler_with_quota(handler, {}, reservation)
            self.assertEqual(next(result), {"data": []})
            result.close()

        auth.refund_image_quota.assert_called_once_with(reservation)
        auth.confirm_image_quota.assert_not_called()

    def test_duplicate_submit_uses_existing_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = 0

            def handler(_payload):
                nonlocal calls
                calls += 1
                time.sleep(0.05)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            first = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            second = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            self.assertEqual(first["id"], "task-1")
            self.assertEqual(second["id"], "task-1")
            task = wait_for_task(service, OWNER, "task-1", "success")
            self.assertEqual(task["data"][0]["url"], "http://example.test/image.png")
            self.assertEqual(calls, 1)

    def test_task_queue_rejects_when_bounded_queue_is_full(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json", task_workers=1, task_queue_size=1)
            payload = {
                "prompt": "cat",
                "model": "gpt-image-2",
                "n": 1,
                "size": None,
                "response_format": "url",
                "base_url": "http://local.test",
            }

            with mock.patch.object(service, "_ensure_workers_started"):
                service._start_task_thread("owner-1:queued-1", "queued-1", "generate", payload, OWNER, "gpt-image-2")
                with self.assertRaisesRegex(RuntimeError, "队列已满"):
                    service._start_task_thread("owner-1:queued-2", "queued-2", "generate", payload, OWNER, "gpt-image-2")

    def test_different_owner_cannot_query_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            service.submit_generation(
                OWNER,
                client_task_id="private-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            wait_for_task(service, OWNER, "private-task", "success")
            result = service.list_tasks(OTHER_OWNER, ["private-task"])

            self.assertEqual(result["items"], [])
            self.assertEqual(result["missing_ids"], ["private-task"])

    def test_success_task_persists_to_new_service_instance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            service = self.make_service(path)
            service.submit_generation(
                OWNER,
                client_task_id="persisted-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            wait_for_task(service, OWNER, "persisted-task", "success")

            reloaded = self.make_service(path)
            result = reloaded.list_tasks(OWNER, ["persisted-task"])

            self.assertEqual(result["missing_ids"], [])
            self.assertEqual(result["items"][0]["status"], "success")
            self.assertEqual(result["items"][0]["data"][0]["url"], "http://example.test/image.png")

    def test_startup_marks_unfinished_tasks_as_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "queued-task",
                                "owner_id": "owner-1",
                                "status": "queued",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                            {
                                "id": "running-task",
                                "owner_id": "owner-1",
                                "status": "running",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            service = self.make_service(path)
            result = service.list_tasks(OWNER, ["queued-task", "running-task"])

            self.assertEqual([item["status"] for item in result["items"]], ["error", "error"])
            self.assertTrue(all("已中断" in item.get("error", "") for item in result["items"]))

    def test_user_generation_task_confirms_reserved_quota_on_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json")
                submitted = service.submit_generation(
                    identity,
                    client_task_id="quota-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                    base_url="http://local.test",
                )
                self.assertEqual(submitted["quota_status"], "reserved")
                task = wait_for_task(service, identity, "quota-task", "success")

            [listed] = auth.list_keys(role="user")
            self.assertEqual(task["quota_status"], "charged")
            self.assertEqual(listed["image_quota"], 0)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_user_generation_task_refunds_reserved_quota_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler=lambda _payload: {"data": []})
                task = service.submit_generation(
                    identity,
                    client_task_id="quota-fail-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                    base_url="http://local.test",
                )
                self.assertEqual(task["quota_status"], "reserved")
                task = wait_for_task(service, identity, "quota-fail-task", "error")

            [listed] = auth.list_keys(role="user")
            self.assertEqual(task["quota_status"], "refunded")
            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_edit_task_uses_existing_quota_reservation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)
            reservation = auth.reserve_image_quota(identity, mode="edit", count=1, reservation_id="pre-read-reservation")

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json")
                submitted = service.submit_edit(
                    identity,
                    client_task_id="edit-quota-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                    base_url="http://local.test",
                    images=[(b"image", "image/png", "image.png")],
                    quota_reservation=reservation,
                )
                self.assertEqual(submitted["quota_status"], "reserved")
                task = wait_for_task(service, identity, "edit-quota-task", "success")

            [listed] = auth.list_keys(role="user")
            self.assertEqual(task["quota_status"], "charged")
            self.assertEqual(listed["image_quota"], 0)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_task_save_failure_refunds_reserved_quota(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json")
                with mock.patch.object(service, "_save_locked", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        service.submit_generation(
                            identity,
                            client_task_id="save-fail-task",
                            prompt="cat",
                            model="gpt-image-2",
                            size=None,
                            base_url="http://local.test",
                        )

            [listed] = auth.list_keys(role="user")
            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_worker_running_save_failure_refunds_reserved_quota(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json")
                payload = {
                    "prompt": "cat",
                    "model": "gpt-image-2",
                    "n": 1,
                    "size": None,
                    "response_format": "url",
                    "base_url": "http://local.test",
                }
                service._submit(identity, client_task_id="running-save-fail-task", mode="generate", payload=payload, start=False)
                with mock.patch.object(service, "_save_locked", side_effect=OSError("disk full")):
                    service._run_task(f"{identity['id']}:running-save-fail-task", "generate", payload, identity, "gpt-image-2")

            [listed] = auth.list_keys(role="user")
            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_worker_final_charge_status_save_failure_does_not_refund_charge(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json")
                payload = {
                    "prompt": "cat",
                    "model": "gpt-image-2",
                    "n": 1,
                    "size": None,
                    "response_format": "url",
                    "base_url": "http://local.test",
                }
                service._submit(identity, client_task_id="final-save-fail-task", mode="generate", payload=payload, start=False)
                with mock.patch.object(service, "_save_locked", side_effect=[None, None, OSError("disk full")]):
                    service._run_task(f"{identity['id']}:final-save-fail-task", "generate", payload, identity, "gpt-image-2")
                task = service.list_tasks(identity, ["final-save-fail-task"])["items"][0]

            [listed] = auth.list_keys(role="user")
            self.assertEqual(task["status"], "success")
            self.assertEqual(task["quota_status"], "charged")
            self.assertEqual(listed["image_quota"], 0)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_worker_confirm_failure_keeps_success_and_refunds_quota(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(Path(tmp_dir) / "image_tasks.json")
                with mock.patch.object(auth, "confirm_image_quota", side_effect=OSError("disk full")):
                    service.submit_generation(
                        identity,
                        client_task_id="confirm-fail-task",
                        prompt="cat",
                        model="gpt-image-2",
                        size=None,
                        base_url="http://local.test",
                    )
                    task = wait_for_task(service, identity, "confirm-fail-task", "success")
                    deadline = time.time() + 1
                    while task.get("quota_status") == "charge_pending" and time.time() < deadline:
                        time.sleep(0.02)
                        task = service.list_tasks(identity, ["confirm-fail-task"])["items"][0]

            [listed] = auth.list_keys(role="user")
            self.assertEqual(task["status"], "success")
            self.assertEqual(task["quota_status"], "refunded")
            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_success_task_with_pending_charge_is_confirmed_on_recovery(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)
            reservation = auth.reserve_image_quota(identity, mode="generate", count=1, reservation_id=f"{identity['id']}:pending-charge-task")
            path.write_text(
                json.dumps({
                    "tasks": [
                        {
                            "id": "pending-charge-task",
                            "owner_id": identity["id"],
                            "status": "success",
                            "mode": "generate",
                            "model": "gpt-image-2",
                            "size": "",
                            "created_at": "2026-01-01 00:00:00",
                            "updated_at": "2026-01-01 00:00:00",
                            "data": [{"url": "http://example.test/image.png"}],
                            "quota_reservation": reservation,
                            "quota_cost": 1,
                            "quota_status": "charge_pending",
                        }
                    ]
                }),
                encoding="utf-8",
            )

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(path)
                task = service.list_tasks(identity, ["pending-charge-task"])["items"][0]

            [listed] = auth.list_keys(role="user")
            [stored_task] = json.loads(path.read_text(encoding="utf-8"))["tasks"]
            self.assertEqual(task["status"], "success")
            self.assertEqual(task["quota_status"], "charged")
            self.assertNotIn("quota_reservation", stored_task)
            self.assertEqual(listed["image_quota"], 0)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_prepared_edit_task_refunds_reserved_quota_on_recovery(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(path)
                prepared = service.prepare_edit(
                    identity,
                    client_task_id="prepared-edit-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                )
                self.assertEqual(prepared["quota_status"], "reserved")
                [listed] = auth.list_keys(role="user")
                self.assertEqual(listed["image_quota_reserved"], 1)

                recovered = self.make_service(path)
                task = recovered.list_tasks(identity, ["prepared-edit-task"])["items"][0]

            [listed] = auth.list_keys(role="user")
            self.assertEqual(task["status"], "error")
            self.assertEqual(task["quota_status"], "refunded")
            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)

    def test_recovery_refund_failure_keeps_reservation_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)
            reservation = auth.reserve_image_quota(
                identity,
                mode="generate",
                count=1,
                reservation_id=f"{identity['id']}:refund-pending-task",
            )
            path.write_text(
                json.dumps({
                    "tasks": [
                        {
                            "id": "refund-pending-task",
                            "owner_id": identity["id"],
                            "status": "queued",
                            "mode": "generate",
                            "model": "gpt-image-2",
                            "created_at": "2026-01-01 00:00:00",
                            "updated_at": "2026-01-01 00:00:00",
                            "quota_reservation": reservation,
                            "quota_cost": 1,
                            "quota_status": "reserved",
                        }
                    ]
                }),
                encoding="utf-8",
            )

            with mock.patch.object(image_task_module, "auth_service", auth):
                with mock.patch.object(auth, "refund_image_quota", side_effect=OSError("disk full")):
                    service = self.make_service(path)
                    task = service.list_tasks(identity, ["refund-pending-task"])["items"][0]

            [listed] = auth.list_keys(role="user")
            [stored_task] = json.loads(path.read_text(encoding="utf-8"))["tasks"]
            self.assertEqual(task["status"], "error")
            self.assertEqual(task["quota_status"], "refund_pending")
            self.assertIn("quota_reservation", stored_task)
            self.assertEqual(listed["image_quota_reserved"], 1)

    def test_prepare_edit_existing_task_reports_not_created(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(path)
                first = service.prepare_edit(
                    identity,
                    client_task_id="same-edit-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                )
                second = service.prepare_edit(
                    identity,
                    client_task_id="same-edit-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                )

            [listed] = auth.list_keys(role="user")
            self.assertTrue(first["_created"])
            self.assertFalse(second["_created"])
            self.assertEqual(listed["image_quota_reserved"], 1)

    def test_fail_prepared_edit_task_refunds_and_clears_reservation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            auth = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            auth._image_quota_multiplier = lambda _mode: Decimal("1")
            _, raw_key = auth.create_key(role="user", name="Alice", image_quota=1)
            identity = auth.authenticate(raw_key)

            with mock.patch.object(image_task_module, "auth_service", auth):
                service = self.make_service(path)
                service.prepare_edit(
                    identity,
                    client_task_id="read-fail-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                )
                failed = service.fail_task(identity, "read-fail-task", "image file is required")

            [listed] = auth.list_keys(role="user")
            payload = json.loads(path.read_text(encoding="utf-8"))
            [stored_task] = payload["tasks"]
            self.assertEqual(failed["status"], "error")
            self.assertEqual(failed["quota_status"], "refunded")
            self.assertNotIn("quota_reservation", stored_task)
            self.assertEqual(listed["image_quota"], 1)
            self.assertEqual(listed["image_quota_reserved"], 0)


if __name__ == "__main__":
    unittest.main()
