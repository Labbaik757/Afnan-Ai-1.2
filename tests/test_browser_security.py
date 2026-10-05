"""Security layer tests: sensitive-action classification, the
human ApprovalGate, controller enforcement, and secret
redaction across records.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.browser import BrowserController
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.security import (
    ApprovalGate,
    SecurityPolicy,
    Sensitivity,
    classify_action,
)
from afnan_ai.redaction import redact_arguments, redact_text, redact_value
from afnan_ai.state import AgentState


# ----------------------------------------------------------------------
# Minimal fake backend with a checkout page
# ----------------------------------------------------------------------

class FakeElement:
    def __init__(self, tag, attrs=None, text="", value=""):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.text = text
        self.value = value
        self.attached = True
        self.clicked = 0

    @property
    def editable(self):
        return self.tag in ("input", "textarea", "select")


class FakePage:
    def __init__(self):
        self.url = "https://shop.example/checkout"
        self.elements = [
            FakeElement("input", {"id": "email", "type": "text",
                                  "placeholder": "Email"}),
            FakeElement("input", {"id": "password", "type": "password"}),
            FakeElement("input", {"id": "card", "type": "text",
                                  "placeholder": "Card number"}),
            FakeElement("button", {"id": "order"}, text="Place Order"),
            FakeElement("button", {"id": "signin"}, text="Sign in"),
            FakeElement("button", {"id": "delete"}, text="Delete account"),
            FakeElement("button", {"id": "send"}, text="Send message"),
            FakeElement("input", {"id": "avatar", "type": "file"}),
        ]


class FakeBackend(BrowserBackend):
    name = "fake-sec"

    def __init__(self):
        self.page = FakePage()
        self.started = False

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False

    def new_page(self):
        return self.page

    def close_page(self, handle):
        pass

    def goto(self, handle, url):
        handle.url = url

    def go_back(self, handle):
        pass

    def go_forward(self, handle):
        pass

    def reload(self, handle):
        pass

    def page_url(self, handle):
        return handle.url

    def page_title(self, handle):
        return "Checkout"

    def query_elements(self, handle, locator, limit):
        sel = locator.get("selector", "")
        text = locator.get("text", "")
        found = []
        for el in handle.elements:
            if sel.startswith("#") and el.attrs.get("id") == sel[1:]:
                found.append(el)
            elif text and text in el.text:
                found.append(el)
        return found[:limit]

    def element_info(self, handle, element):
        return {
            "tag": element.tag,
            "text": element.text,
            "attributes": dict(element.attrs),
            "visible": True,
            "enabled": True,
            "editable": element.editable,
            "value": element.value,
        }

    def click_element(self, handle, element, timeout_ms):
        element.clicked += 1

    def fill_element(self, handle, element, text, timeout_ms):
        element.value = text

    def clear_element(self, handle, element, timeout_ms):
        element.value = ""

    def select_option(self, handle, element, value, timeout_ms):
        element.value = value

    def press_key(self, handle, element, key, timeout_ms):
        pass

    def scroll_page(self, handle, dx, dy):
        pass

    def scroll_to_element(self, handle, element):
        pass

    def set_input_files(self, handle, element, path, timeout_ms):
        element.value = path

    def page_text(self, handle):
        return "Checkout page"

    def interactive_elements(self, handle, limit):
        return list(handle.elements)[:limit]

    def screenshot(self, handle):
        return b"\x89PNG fake"


def make_controller(approver=None, policy=None):
    backend = FakeBackend()
    gate = ApprovalGate(policy=policy, approver=approver)
    controller = BrowserController(backend=backend, approval_gate=gate)
    controller.launch()
    controller.new_tab("https://shop.example/checkout")
    return controller, backend


def el(backend, element_id):
    return next(
        e for e in backend.page.elements if e.attrs.get("id") == element_id
    )


def element_dict(text, tag="button", attrs=None):
    return {"tag": tag, "text": text, "attributes": attrs or {}}


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

class TestClassification(unittest.TestCase):
    def test_purchase_buttons(self):
        for text in ("Place Order", "Buy Now", "Checkout", "Pay"):
            risk = classify_action(
                "browser_click", element=element_dict(text)
            )
            self.assertTrue(risk.sensitive, text)
            self.assertEqual(risk.category, "purchase")

    def test_send_and_destructive(self):
        risk = classify_action(
            "browser_click", element=element_dict("Send message")
        )
        self.assertEqual(risk.category, "message_send")
        risk = classify_action(
            "browser_click", element=element_dict("Delete account")
        )
        self.assertEqual(risk.category, "destructive")
        risk = classify_action(
            "browser_click", element=element_dict("Send email now")
        )
        self.assertEqual(risk.category, "email_send")

    def test_safe_actions_not_flagged(self):
        for text in ("Sign in", "Continue", "Search", "Help", "Close"):
            risk = classify_action(
                "browser_click", element=element_dict(text)
            )
            self.assertEqual(
                risk.sensitivity, Sensitivity.SAFE, text
            )

    def test_password_and_payment_fields(self):
        risk = classify_action(
            "browser_type",
            element=element_dict("", "input", {"type": "password"}),
        )
        self.assertEqual(risk.category, "credential_input")
        risk = classify_action(
            "browser_type",
            element=element_dict(
                "", "input", {"type": "text", "placeholder": "Card number"}
            ),
        )
        self.assertEqual(risk.category, "payment_input")

    def test_upload_always_sensitive(self):
        risk = classify_action("browser_upload_file")
        self.assertTrue(risk.sensitive)
        self.assertEqual(risk.category, "file_upload")


# ----------------------------------------------------------------------
# Gate behaviour
# ----------------------------------------------------------------------

class TestApprovalGate(unittest.TestCase):
    def _risk(self, category="purchase"):
        from afnan_ai.browser.security import ActionRisk
        return ActionRisk(Sensitivity.SENSITIVE, category, "test")

    def test_no_approver_means_no_execution(self):
        gate = ApprovalGate()
        decision = gate.check(self._risk(), tool_name="browser_click")
        self.assertFalse(decision.allowed)

    def test_human_yes_and_no(self):
        gate = ApprovalGate(approver=lambda req: True)
        self.assertTrue(
            gate.check(self._risk(), tool_name="browser_click").allowed
        )
        gate = ApprovalGate(approver=lambda req: False)
        self.assertFalse(
            gate.check(self._risk(), tool_name="browser_click").allowed
        )

    def test_blocked_category_never_runs(self):
        policy = SecurityPolicy(
            blocked_categories=frozenset({"purchase"})
        )
        gate = ApprovalGate(policy=policy, approver=lambda req: True)
        self.assertFalse(
            gate.check(self._risk(), tool_name="browser_click").allowed
        )

    def test_credential_input_allowed_but_classified(self):
        gate = ApprovalGate()  # no approver at all
        decision = gate.check(
            self._risk("credential_input"), tool_name="browser_type"
        )
        self.assertTrue(decision.allowed)

    def test_broken_approver_denies(self):
        def boom(req):
            raise RuntimeError("ui gone")
        gate = ApprovalGate(approver=boom)
        self.assertFalse(
            gate.check(self._risk(), tool_name="browser_click").allowed
        )

    def test_request_arguments_are_redacted(self):
        seen = {}

        def approver(req):
            seen.update(req.to_dict())
            return True

        gate = ApprovalGate(approver=approver)
        gate.check(
            self._risk(),
            tool_name="browser_type",
            arguments={"selector": "#password", "text": "hunter2"},
        )
        self.assertNotIn("hunter2", str(seen))

    def test_decisions_audited(self):
        gate = ApprovalGate()
        gate.check(self._risk(), tool_name="browser_click")
        self.assertEqual(len(gate.decisions), 1)
        self.assertFalse(gate.decisions[0]["allowed"])


# ----------------------------------------------------------------------
# Controller enforcement
# ----------------------------------------------------------------------

class TestControllerEnforcement(unittest.TestCase):
    def test_purchase_blocked_without_approver(self):
        controller, backend = make_controller()
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"selector": "#order"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.APPROVAL_REQUIRED
        )
        self.assertEqual(el(backend, "order").clicked, 0)

    def test_purchase_runs_with_approval(self):
        controller, backend = make_controller(approver=lambda req: True)
        result = controller.click({"selector": "#order"})
        self.assertEqual(el(backend, "order").clicked, 1)
        self.assertEqual(
            result["sensitivity"]["category"], "purchase"
        )

    def test_purchase_denied_by_human(self):
        controller, backend = make_controller(approver=lambda req: False)
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"selector": "#order"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.APPROVAL_DENIED
        )
        self.assertEqual(el(backend, "order").clicked, 0)

    def test_safe_click_needs_no_approval(self):
        controller, backend = make_controller()
        controller.click({"selector": "#signin"})
        self.assertEqual(el(backend, "signin").clicked, 1)

    def test_upload_gated(self):
        import tempfile, os
        controller, backend = make_controller()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False
        ) as f:
            f.write("hello")
            path = f.name
        try:
            with self.assertRaises(BrowserException) as ctx:
                controller.upload_file({"selector": "#avatar"}, path)
            self.assertEqual(
                ctx.exception.code, BrowserErrorCode.APPROVAL_REQUIRED
            )
            self.assertEqual(el(backend, "avatar").value, "")
        finally:
            os.unlink(path)


# ----------------------------------------------------------------------
# Redaction
# ----------------------------------------------------------------------

class TestRedaction(unittest.TestCase):
    def test_nested_secret_keys(self):
        data = {
            "user": "afnan",
            "credentials": {"password": "hunter2", "api_token": "abc123"},
            "note": "ok",
        }
        out = redact_value(data)
        self.assertNotIn("hunter2", str(out))
        self.assertNotIn("abc123", str(out))
        self.assertEqual(out["user"], "afnan")

    def test_text_patterns(self):
        self.assertNotIn(
            "4111", redact_text("card 4111 1111 1111 1111 charged")
        )
        self.assertNotIn("sekret", redact_text("Authorization: Bearer sekret"))
        self.assertNotIn("hunter2", redact_text("password=hunter2&x=1"))
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature"
        self.assertNotIn("eyJ", redact_text(f"token: {jwt}"))

    def test_password_field_arguments(self):
        out = redact_arguments(
            {"selector": "#password", "text": "hunter2"}
        )
        self.assertEqual(out["text"], "***")
        out = redact_arguments(
            {"selector": "#username", "text": "afnan"}
        )
        self.assertEqual(out["text"], "afnan")

    def test_state_records_are_redacted(self):
        state = AgentState.create("test")
        state.add_tool_result(
            "some_tool", success=True,
            output={"api_token": "abc123", "value": "kept"},
            error=None,
        )
        blob = state.to_json()
        self.assertNotIn("abc123", blob)
        self.assertIn("kept", blob)

    def test_password_value_masked_in_controller_results(self):
        controller, backend = make_controller()
        result = controller.type_text(
            {"selector": "#password"}, "hunter2"
        )
        self.assertEqual(result["value"], "***")
        found = controller.find_elements({"selector": "#password"})
        self.assertEqual(found[0]["value"], "***")
        # the field itself really holds the secret (execution
        # uses real values; only records are masked)
        self.assertEqual(el(backend, "password").value, "hunter2")


if __name__ == "__main__":
    unittest.main()
