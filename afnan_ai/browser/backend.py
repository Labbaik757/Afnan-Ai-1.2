"""Playwright engine adapter: Chromium via Playwright.

This module holds the one concrete :class:`BrowserEngineAdapter`
shipping today: :class:`PlaywrightAdapter` (Chromium / Chrome /
Edge / Firefox / WebKit through Microsoft's Playwright, the same
on Windows, macOS and Linux).  The BrowserController never talks
to it directly any more — the :class:`AfnanBrowserRuntime` owns
browser lifecycle and state and routes everything through the
adapter interface in ``engine.py``, so this adapter can later be
replaced by an Afnan Chromium adapter without touching the
runtime, controller, tools or agent.

Playwright is imported lazily inside :meth:`PlaywrightAdapter.start`,
so constructing the adapter — or importing this module — never
requires it; if it (or its browser binaries) is missing, ``start``
raises a structured ``browser_unavailable`` error instead of
crashing.  ``BrowserBackend`` / ``PlaywrightBackend`` remain as
compatibility aliases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.engine import BrowserEngineAdapter
from afnan_ai.redaction import redact_text

#: Browser names the controller accepts (mapped by the backend)
SUPPORTED_BROWSERS = ("chromium", "chrome", "edge", "firefox", "webkit")


class PlaywrightAdapter(BrowserEngineAdapter):
    """Real browser control via Playwright (sync API).

    Works on Windows, macOS and Linux.  Requires the ``playwright``
    package and its browser binaries::

        pip install playwright
        playwright install chromium
    """

    name = "playwright"

    #: our browser name -> (playwright browser type, channel)
    _BROWSER_MAP = {
        "chromium": ("chromium", None),
        "chrome": ("chromium", "chrome"),
        "edge": ("chromium", "msedge"),
        "firefox": ("firefox", None),
        "webkit": ("webkit", None),
    }

    def __init__(self, downloads_dir: str | None = None):
        self._playwright = None
        self._browser = None
        self._browser_type = None
        self._headless = True
        self._context = None
        self._download_records: list[dict[str, Any]] = []
        self._downloads_dir = downloads_dir
        # isolated profile contexts (own cookies/storage each)
        self._contexts: dict[str, Any] = {}
        self._active_profile: str = "default"
        # per-page network events, keyed by page object id
        self._network: dict[int, list[dict[str, Any]]] = {}
        self._dialogs: dict[int, list[dict[str, Any]]] = {}

    # -- lifecycle ------------------------------------------------------
    def start(self, browser: str, headless: bool) -> None:
        sync_playwright = self._import_playwright()
        browser_type_name, channel = self._BROWSER_MAP.get(
            browser, ("chromium", None)
        )
        try:
            self._playwright = sync_playwright().start()
            browser_type = getattr(self._playwright, browser_type_name)
            self._browser_type = browser_type
            self._headless = headless
            launch_kwargs: dict[str, Any] = {"headless": headless}
            if channel:
                launch_kwargs["channel"] = channel
            self._browser = browser_type.launch(**launch_kwargs)
            self._context = self._browser.new_context()
            self._contexts = {"default": self._context}
            self._active_profile = "default"
        except BrowserException:
            self._cleanup()
            raise
        except Exception as e:
            self._cleanup()
            raise BrowserException(
                f"Could not launch browser {browser!r}: {e}. "
                "Make sure the browser binaries are installed "
                "(`playwright install`).",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
                details={"browser": browser},
            ) from e

    def connect(self, endpoint: str) -> None:
        sync_playwright = self._import_playwright()
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.connect_over_cdp(endpoint)
            contexts = self._browser.contexts
            self._context = contexts[0] if contexts else self._browser.new_context()
            self._contexts = {"default": self._context}
            self._active_profile = "default"
        except BrowserException:
            self._cleanup()
            raise
        except Exception as e:
            self._cleanup()
            raise BrowserException(
                f"Could not connect to browser at {endpoint!r}: {e}",
                code=BrowserErrorCode.CONNECTION_FAILED,
                details={"endpoint": endpoint},
            ) from e

    def stop(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._context = None
        self._playwright = None
        self._contexts = {}
        self._active_profile = "default"
        self._network = {}
        self._dialogs = {}

    @staticmethod
    def _import_playwright():
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            raise BrowserException(
                "Playwright is not installed. Install it with "
                "`pip install playwright` and "
                "`playwright install chromium` to enable browser "
                "control.",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
            ) from e
        return sync_playwright

    def _require_page(self):
        if self._context is None:
            raise BrowserException(
                "Browser is not running",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )
        return self._context

    # -- pages ------------------------------------------------------------
    def new_page(self) -> Any:
        page = self._require_page().new_page()
        try:
            page.on("download", self._on_download)
        except Exception:
            pass
        self._track_network(page)
        self._track_dialogs(page)
        return page

    def _track_dialogs(self, page: Any) -> None:
        """Record dialogs and dismiss them — a page must never
        hang the agent waiting on an alert nobody can see.  The
        dialog text is redacted; nothing is ever auto-confirmed
        (dismiss is the safe default for confirm/prompt too)."""
        seen: list[dict[str, Any]] = []
        self._dialogs[id(page)] = seen

        def on_dialog(dialog: Any) -> None:
            try:
                seen.append({
                    "type": getattr(dialog, "type", "alert"),
                    "message": redact_text(
                        str(getattr(dialog, "message", "") or "")
                    )[:200],
                    "action": "dismissed",
                })
            finally:
                try:
                    dialog.dismiss()
                except Exception:
                    pass

        try:
            page.on("dialog", on_dialog)
        except Exception:
            pass

    def dialogs(self, handle: Any) -> list[dict[str, Any]]:
        return [dict(d) for d in self._dialogs.get(id(handle), [])]

    # -- network awareness (observation only) ---------------------------

    def _track_network(self, page: Any) -> None:
        events: list[dict[str, Any]] = []
        self._network[id(page)] = events

        def record(entry: dict[str, Any]) -> None:
            events.append(entry)
            if len(events) > 200:  # bounded history per page
                del events[:-200]

        def on_failed(request: Any) -> None:
            record({
                "url": request.url,
                "method": request.method,
                "status": None,
                "failure": request.failure,
                "resource_type": request.resource_type,
            })

        def on_response(response: Any) -> None:
            if response.status >= 400:
                record({
                    "url": response.url,
                    "method": response.request.method,
                    "status": response.status,
                    "failure": None,
                    "resource_type": response.request.resource_type,
                })

        try:
            page.on("requestfailed", on_failed)
            page.on("response", on_response)
        except Exception:
            pass

    def network_events(self, handle: Any) -> list[dict[str, Any]]:
        return [
            dict(e) for e in self._network.get(id(handle), [])
        ]

    # -- profiles (isolated contexts) -------------------------------------

    #: preference keys a profile may carry into a new context
    _PROFILE_CONTEXT_KEYS = (
        "user_agent", "locale", "timezone_id", "color_scheme",
        "viewport",
    )

    def create_profile_context(
        self, name: str, options: dict[str, Any]
    ) -> str:
        if self._browser is None:
            raise BrowserException(
                "Browser is not running",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )
        if name in self._contexts:
            return name
        kwargs = {
            key: options[key]
            for key in self._PROFILE_CONTEXT_KEYS
            if options.get(key) is not None
        }
        try:
            user_data_dir = options.get("user_data_dir")
            if user_data_dir:
                # Persistent profile: cookies/storage live in
                # the profile's own directory on disk and
                # survive restarts.
                self._contexts[name] = (
                    self._browser_type.launch_persistent_context(
                        str(user_data_dir),
                        headless=self._headless,
                        **kwargs,
                    )
                )
            else:
                self._contexts[name] = self._browser.new_context(
                    **kwargs
                )
        except Exception as e:
            raise BrowserException(
                f"Could not create browser profile {name!r}: {e}",
                code=BrowserErrorCode.PROFILE_ERROR,
                details={"profile": name},
            ) from e
        return name

    def is_alive(self) -> bool:
        try:
            if self._context is None:
                return False
            if self._browser is not None:
                return bool(self._browser.is_connected())
            return True
        except Exception:
            return False

    def capabilities(self) -> dict[str, bool]:
        return {
            "tabs": True,
            "accessibility": True,
            "screenshots": True,
            "downloads": True,
            "uploads": True,
            "persistent_profiles": True,
            "iframe": True,
            "shadow_dom": True,
            "network_observation": True,
            "dialogs": True,
        }

    def set_active_profile(self, name: str) -> None:
        context = self._contexts.get(name)
        if context is None:
            raise BrowserException(
                f"Unknown browser profile {name!r}",
                code=BrowserErrorCode.PROFILE_ERROR,
                details={"profile": name},
            )
        self._active_profile = name
        self._context = context

    def _on_download(self, download: Any) -> None:
        """Record a browser download; it is saved, never opened."""
        record: dict[str, Any] = {
            "id": f"dl_{len(self._download_records) + 1}",
            "url": getattr(download, "url", "") or "",
            "filename": (
                getattr(download, "suggested_filename", "") or "download"
            ),
            "state": "started",
            "path": None,
            "failure": None,
        }
        self._download_records.append(record)
        try:
            if self._downloads_dir:
                target_dir = Path(self._downloads_dir)
            else:
                target_dir = Path.home() / "Downloads" / "afnan-ai"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / record["filename"]
            download.save_as(str(target))
            record["path"] = str(target)
            record["state"] = "completed"
        except Exception as exc:  # download failed mid-flight
            record["state"] = "failed"
            record["failure"] = str(exc)

    def downloads(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._download_records]

    def close_page(self, handle: Any) -> None:
        handle.close()

    def goto(self, handle: Any, url: str) -> None:
        handle.goto(url)

    def go_back(self, handle: Any) -> None:
        handle.go_back()

    def go_forward(self, handle: Any) -> None:
        handle.go_forward()

    def reload(self, handle: Any) -> None:
        handle.reload()

    def page_url(self, handle: Any) -> str:
        return handle.url or ""

    def page_title(self, handle: Any) -> str:
        return handle.title() or ""

    # -- element interaction (Playwright) --------------------------------
    # Locator strategies, most stable/specific first.  A locator
    # dict may carry several; the first that matches anything wins.
    _LOCATOR_ORDER = ("selector", "test_id", "label", "placeholder", "role", "text")

    def _locators(self, page, locator: dict[str, Any]):
        # When a frame is named, look inside it (iframes); CSS
        # selectors also pierce open shadow DOM by default in
        # Playwright, so shadow content needs no special casing.
        root = page
        if locator.get("frame"):
            root = page.frame_locator(str(locator["frame"]))
        candidates = []
        for key in self._LOCATOR_ORDER:
            if locator.get(key) is None:
                continue
            if key == "selector":
                candidates.append(root.locator(str(locator["selector"])))
            elif key == "test_id":
                candidates.append(root.get_by_test_id(str(locator["test_id"])))
            elif key == "label":
                candidates.append(root.get_by_label(str(locator["label"])))
            elif key == "placeholder":
                candidates.append(
                    root.get_by_placeholder(str(locator["placeholder"]))
                )
            elif key == "role":
                kwargs = {}
                if locator.get("name"):
                    kwargs["name"] = str(locator["name"])
                candidates.append(
                    root.get_by_role(str(locator["role"]), **kwargs)
                )
            elif key == "text":
                candidates.append(root.get_by_text(str(locator["text"])))
        return candidates

    def query_elements(
        self, handle: Any, locator: dict[str, Any], limit: int
    ) -> list[Any]:
        for candidate in self._locators(handle, locator):
            try:
                count = candidate.count()
            except Exception:
                continue
            if count:
                handles = []
                for i in range(min(count, max(limit, 1))):
                    element_handle = candidate.nth(i).element_handle()
                    if element_handle is not None:
                        handles.append(element_handle)
                if handles:
                    return handles
        return []

    def element_info(self, handle: Any, element: Any) -> dict[str, Any]:
        try:
            data = element.evaluate(
                """el => {
                    const attrs = {};
                    for (const name of ['id','name','type','role','aria-label',
                                        'placeholder','href','value','title']) {
                        const v = el.getAttribute(name);
                        if (v !== null) attrs[name] = v;
                    }
                    const text = (el.innerText || el.textContent || '')
                        .trim().replace(/\\s+/g, ' ').slice(0, 200);
                    const editable = !!el.isContentEditable ||
                        ['INPUT','TEXTAREA','SELECT'].includes(el.tagName);
                    return {
                        tag: el.tagName.toLowerCase(),
                        text: text,
                        attributes: attrs,
                        editable: editable,
                        value: (el.value !== undefined && el.value !== null)
                            ? String(el.value).slice(0, 200) : ''
                    };
                }"""
            )
            data["visible"] = bool(element.is_visible())
            data["enabled"] = bool(element.is_enabled())
            return data
        except Exception as e:
            raise self._driver_error(e, "inspect element") from e

    def click_element(self, handle: Any, element: Any, timeout_ms: int) -> None:
        try:
            element.click(timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "click element") from e

    def fill_element(
        self, handle: Any, element: Any, text: str, timeout_ms: int
    ) -> None:
        try:
            element.fill(text, timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "type into element") from e

    def clear_element(self, handle: Any, element: Any, timeout_ms: int) -> None:
        try:
            element.fill("", timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "clear element") from e

    def set_input_files(
        self, handle: Any, element: Any, path: str, timeout_ms: int
    ) -> None:
        try:
            element.set_input_files(path, timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "upload file") from e

    def select_option(
        self, handle: Any, element: Any, value: str, timeout_ms: int
    ) -> None:
        try:
            element.select_option(value=value, timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "select option") from e

    def press_key(
        self, handle: Any, element: Any | None, key: str, timeout_ms: int
    ) -> None:
        try:
            if element is not None:
                element.press(key, timeout=timeout_ms)
            else:
                handle.keyboard.press(key)
        except Exception as e:
            raise self._driver_error(e, "press key") from e

    def scroll_page(self, handle: Any, dx: int, dy: int) -> None:
        try:
            handle.mouse.wheel(dx, dy)
        except Exception as e:
            raise self._driver_error(e, "scroll page") from e

    def scroll_to_element(self, handle: Any, element: Any) -> None:
        try:
            element.scroll_into_view_if_needed()
        except Exception as e:
            raise self._driver_error(e, "scroll to element") from e

    # -- page observation --------------------------------------------------
    def page_text(self, handle: Any) -> str:
        try:
            return handle.inner_text("body", timeout=3000) or ""
        except Exception as e:
            raise self._driver_error(e, "read page text") from e

    def accessibility_snapshot(self, handle: Any) -> Any:
        """The page's accessibility tree via the modern ARIA API.

        Uses ``page.aria_snapshot()`` (current Playwright) and
        parses its YAML into the normalized tree shape; the
        deprecated ``page.accessibility`` API is only a fallback
        for very old Playwright versions, and a driver with
        neither raises a structured error so the controller can
        fall back to the DOM.
        """
        aria_text = None
        try:
            aria_text = handle.aria_snapshot()
        except AttributeError:
            aria_text = None
        except Exception as e:
            raise self._driver_error(
                e, "accessibility snapshot"
            ) from e
        if aria_text:
            from afnan_ai.browser.accessibility import (
                parse_aria_snapshot,
            )

            return parse_aria_snapshot(aria_text)
        try:
            accessibility = handle.accessibility
        except AttributeError as e:
            raise BrowserException(
                "This browser driver does not expose an "
                "accessibility tree",
                code=BrowserErrorCode.OPERATION_FAILED,
            ) from e
        try:
            return accessibility.snapshot() or {}
        except Exception as e:
            raise self._driver_error(e, "accessibility snapshot") from e

    _CONTENT_JS = """() => {
      const clean = (el) => ((el && el.innerText) || '')
        .replace(/\\s+/g, ' ').trim();
      const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')]
        .map(h => ({level: parseInt(h.tagName[1], 10), text: clean(h)}))
        .filter(h => h.text);
      const paragraphs = [...document.querySelectorAll('p')]
        .map(p => clean(p)).filter(Boolean);
      const lists = [...document.querySelectorAll('ul, ol')]
        .map(l => [...l.querySelectorAll(':scope > li')]
          .map(li => clean(li)).filter(Boolean))
        .filter(l => l.length);
      const links = [...document.querySelectorAll('a[href]')]
        .map(a => ({text: clean(a), url: a.href}))
        .filter(l => l.url && !l.url.startsWith('javascript:'));
      const tables = [...document.querySelectorAll('table')]
        .map(t => ({
          caption: clean(t.querySelector('caption')),
          rows: [...t.querySelectorAll('tr')]
            .map(tr => [...tr.querySelectorAll('th, td')]
              .map(c => clean(c)))
        }))
        .filter(t => t.rows.length);
      return {
        title: document.title || '',
        url: location.href,
        headings, paragraphs, lists, links, tables,
        text: clean(document.body)
      };
    }"""

    def page_content(self, handle: Any) -> dict[str, Any]:
        try:
            result = handle.evaluate(self._CONTENT_JS)
        except Exception as e:
            raise self._driver_error(e, "page content extraction") from e
        return result if isinstance(result, dict) else {}

    _PROBE_JS = """() => {
      const frameworks = [];
      if (window.__NEXT_DATA__) frameworks.push('nextjs');
      if (window.React || document.querySelector(
        '[data-reactroot], [data-reactid]')) frameworks.push('react');
      if (window.Vue || document.querySelector('[data-v-app]'))
        frameworks.push('vue');
      if (window.ng) frameworks.push('angular');
      const body = document.body;
      const text = body ? (body.innerText || '') : '';
      const html = body ? body.innerHTML : '';
      let hash = 0;
      for (let i = 0; i < html.length; i++) {
        hash = ((hash << 5) - hash + html.charCodeAt(i)) | 0;
      }
      return {
        url: location.href,
        title: document.title,
        ready_state: document.readyState,
        frameworks: frameworks,
        text_length: text.length,
        element_count: document.querySelectorAll('*').length,
        content_hash: String(hash)
      };
    }"""

    def page_probe(self, handle: Any) -> dict[str, Any]:
        try:
            result = handle.evaluate(self._PROBE_JS)
        except Exception as e:
            raise self._driver_error(e, "page probe") from e
        return result if isinstance(result, dict) else {}

    def interactive_elements(
        self, handle: Any, limit: int
    ) -> list[Any]:
        selector = (
            "a, button, input, select, textarea, "
            "[role='button'], [role='link'], [role='textbox'], "
            "[contenteditable='true']"
        )
        try:
            return list(handle.query_selector_all(selector))[:limit]
        except Exception as e:
            raise self._driver_error(e, "list interactive elements") from e

    def list_pages(self) -> list[Any]:
        if self._context is None:
            return []
        try:
            return list(self._context.pages)
        except Exception:
            return []

    def wait_for(
        self, handle: Any, spec: dict[str, Any], timeout_ms: int
    ) -> None:
        kind = spec.get("kind")
        try:
            if kind in ("element_present", "element_hidden"):
                candidates = self._locators(handle, spec.get("locator") or {})
                if not candidates:
                    raise BrowserException(
                        "No locator strategy to wait for",
                        code=BrowserErrorCode.INVALID_LOCATOR,
                    )
                chosen = candidates[0]
                for candidate in candidates:
                    if candidate.count():
                        chosen = candidate
                        break
                state = "attached" if kind == "element_present" else "hidden"
                chosen.first.wait_for(state=state, timeout=timeout_ms)
            elif kind == "text_present":
                handle.get_by_text(str(spec["text"])).first.wait_for(
                    state="attached", timeout=timeout_ms
                )
            elif kind == "url_contains":
                value = str(spec["value"])
                handle.wait_for_url(
                    lambda url: value in (url or ""), timeout=timeout_ms
                )
            elif kind == "title_contains":
                handle.wait_for_function(
                    "v => document.title.includes(v)",
                    arg=str(spec["value"]),
                    timeout=timeout_ms,
                )
            else:
                raise BrowserException(
                    f"Unknown wait condition {kind!r}",
                    code=BrowserErrorCode.INVALID_LOCATOR,
                )
        except BrowserException:
            raise
        except Exception as e:
            raise self._driver_error(e, f"wait for {kind}") from e

    def screenshot(self, handle: Any) -> bytes:
        try:
            return handle.screenshot()
        except Exception as e:
            raise self._driver_error(e, "take a screenshot") from e

    # -- computer-use primitives (Playwright mouse/DOM) ----------------
    def element_box(
        self, handle: Any, element: Any
    ) -> dict[str, Any] | None:
        try:
            box = element.bounding_box()
        except Exception as e:
            raise self._driver_error(e, "read element bounds") from e
        if not box:
            return None
        return {
            "x": float(box["x"]), "y": float(box["y"]),
            "width": float(box["width"]), "height": float(box["height"]),
        }

    def mouse_click(
        self, handle: Any, x: float, y: float, click_count: int = 1
    ) -> None:
        try:
            handle.mouse.click(x, y, click_count=max(1, int(click_count)))
        except Exception as e:
            raise self._driver_error(e, "click at coordinates") from e

    def mouse_move(self, handle: Any, x: float, y: float) -> None:
        try:
            handle.mouse.move(x, y)
        except Exception as e:
            raise self._driver_error(e, "move the mouse") from e

    def mouse_drag(
        self, handle: Any, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        try:
            handle.mouse.move(x1, y1)
            handle.mouse.down()
            handle.mouse.move(x2, y2)
            handle.mouse.up()
        except Exception as e:
            raise self._driver_error(e, "drag the mouse") from e

    def focus_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        try:
            element.focus(timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "focus element") from e

    def set_checked(
        self, handle: Any, element: Any, checked: bool, timeout_ms: int
    ) -> None:
        try:
            element.set_checked(bool(checked), timeout=timeout_ms)
        except Exception as e:
            raise self._driver_error(e, "set element checked state") from e

    @staticmethod
    def _driver_error(e: Exception, context: str) -> BrowserException:
        """Map a raw Playwright failure to a structured error."""
        name = type(e).__name__
        message = str(e)
        if name == "TimeoutError" or "timeout" in message.lower():
            return BrowserException(
                f"Timed out while trying to {context}: {message}",
                code=BrowserErrorCode.TIMEOUT,
            )
        lowered = message.lower()
        if "not attached" in lowered or "detached" in lowered:
            return BrowserException(
                f"Element is no longer attached to the page "
                f"({context}): {message}",
                code=BrowserErrorCode.STALE_ELEMENT,
            )
        return BrowserException(
            f"Browser failed to {context}: {message}",
            code=BrowserErrorCode.OPERATION_FAILED,
        )


#: Compatibility aliases — the pre-runtime names keep working.
BrowserBackend = BrowserEngineAdapter
PlaywrightBackend = PlaywrightAdapter
