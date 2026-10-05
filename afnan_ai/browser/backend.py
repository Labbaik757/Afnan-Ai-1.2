"""Browser backends: the driver boundary of the BrowserController.

A *backend* is the thin, browser-specific layer that actually
talks to a browser process.  The :class:`BrowserController` owns
sessions, tabs and validation and only ever calls this interface,
so the browser technology can be swapped (or faked in tests)
without touching controller, tool or agent code.

The shipped backend is :class:`PlaywrightBackend` (Chromium /
Chrome / Edge / Firefox / WebKit through Microsoft's Playwright,
which works the same on Windows, macOS and Linux).  Playwright is
imported lazily inside :meth:`PlaywrightBackend.start`, so merely
constructing the backend — or importing this module — never
requires it; if it (or its browser binaries) is missing, ``start``
raises a structured ``browser_unavailable`` error instead of
crashing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.redaction import redact_text

#: Browser names the controller accepts (mapped by the backend)
SUPPORTED_BROWSERS = ("chromium", "chrome", "edge", "firefox", "webkit")


class BrowserBackend(ABC):
    """Interface a concrete browser driver implements.

    Handles are opaque backend objects (a Playwright Page, a fake
    test page, ...).  Implementations raise BrowserException for
    expected failures; the controller also wraps unexpected
    exceptions into structured errors, so a backend never leaks a
    bare driver exception through the controller.
    """

    name: str = "backend"

    @abstractmethod
    def start(self, browser: str, headless: bool) -> None:
        """Launch a browser process (or raise BrowserException)."""

    @abstractmethod
    def connect(self, endpoint: str) -> None:
        """Connect to an already-running browser (CDP endpoint)."""

    @abstractmethod
    def new_page(self) -> Any:
        """Open a new tab/page and return its handle."""

    @abstractmethod
    def close_page(self, handle: Any) -> None:
        """Close one tab/page."""

    @abstractmethod
    def goto(self, handle: Any, url: str) -> None:
        """Navigate a tab to *url*."""

    @abstractmethod
    def go_back(self, handle: Any) -> None:
        """Go back in a tab's history."""

    @abstractmethod
    def go_forward(self, handle: Any) -> None:
        """Go forward in a tab's history."""

    @abstractmethod
    def reload(self, handle: Any) -> None:
        """Reload a tab."""

    @abstractmethod
    def page_url(self, handle: Any) -> str:
        """Return the tab's current URL."""

    @abstractmethod
    def page_title(self, handle: Any) -> str:
        """Return the tab's current page title."""

    @abstractmethod
    def stop(self) -> None:
        """Close the browser and release the driver."""

    # -- element interaction -------------------------------------------------
    # These have safe defaults (a structured "not supported" error)
    # so simple backends/fakes stay valid; real backends override
    # them.  ``locator`` is a dict of lookup strategies, e.g.
    # {"selector": "#login"} or {"role": "button", "name": "Sign in"}
    # or {"text": "Sign in"} — most specific first.
    def _unsupported(self, operation: str):
        raise BrowserException(
            f"Browser backend {self.name!r} does not support {operation}",
            code=BrowserErrorCode.OPERATION_FAILED,
        )

    def query_elements(
        self, handle: Any, locator: dict[str, Any], limit: int
    ) -> list[Any]:
        self._unsupported("element lookup")

    def element_info(self, handle: Any, element: Any) -> dict[str, Any]:
        self._unsupported("element inspection")

    def click_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._unsupported("clicking")

    def fill_element(
        self, handle: Any, element: Any, text: str, timeout_ms: int
    ) -> None:
        self._unsupported("typing")

    def clear_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._unsupported("clearing")

    def set_input_files(
        self, handle: Any, element: Any, path: str, timeout_ms: int
    ) -> None:
        self._unsupported("file upload")

    def select_option(
        self, handle: Any, element: Any, value: str, timeout_ms: int
    ) -> None:
        self._unsupported("selecting an option")

    def press_key(
        self,
        handle: Any,
        element: Any | None,
        key: str,
        timeout_ms: int,
    ) -> None:
        self._unsupported("keyboard actions")

    def scroll_page(self, handle: Any, dx: int, dy: int) -> None:
        self._unsupported("scrolling")

    def scroll_to_element(self, handle: Any, element: Any) -> None:
        self._unsupported("scrolling to an element")

    # -- page observation --------------------------------------------------
    def page_text(self, handle: Any) -> str:
        """Return the page's visible text."""
        self._unsupported("reading page text")

    def accessibility_snapshot(self, handle: Any) -> Any:
        """The browser's accessibility tree (driver-native).

        Backends without one raise via ``_unsupported`` and the
        controller derives a tree from the DOM instead.
        """
        self._unsupported("accessibility snapshot")

    def page_content(self, handle: Any) -> dict[str, Any]:
        """Structured page document (headings, paragraphs, lists,
        links, tables) for content extraction."""
        self._unsupported("page content extraction")

    def page_probe(self, handle: Any) -> dict[str, Any]:
        """Dynamic-state probe (SPA awareness): readiness, detected
        frameworks, text/element counts and a content hash."""
        self._unsupported("page probe")

    def downloads(self) -> list[dict[str, Any]]:
        """Downloads the driver has seen (empty by default)."""
        return []

    def network_events(self, handle: Any) -> list[dict[str, Any]]:
        """Requests the page made (failures / HTTP errors).

        Observation only — there is deliberately no backend API
        for firing arbitrary network requests.
        """
        return []

    def dialogs(self, handle: Any) -> list[dict[str, Any]]:
        """Dialogs (alert/confirm/prompt) the page raised.

        Observation only: the driver records and dismisses them
        so a page can never hang the agent; the controller only
        reports what happened.
        """
        return []

    def create_profile_context(
        self, name: str, options: dict[str, Any]
    ) -> str:
        """Create an isolated profile context (own cookies/storage)."""
        self._unsupported("profile contexts")

    def set_active_profile(self, name: str) -> None:
        """Make the named profile context the active one."""
        self._unsupported("profile contexts")

    def interactive_elements(
        self, handle: Any, limit: int
    ) -> list[Any]:
        """Return handles of the page's interactive elements."""
        self._unsupported("listing interactive elements")

    def list_pages(self) -> list[Any]:
        """All page handles the driver knows, including popups the
        controller did not create.  Default: none tracked."""
        return []

    def wait_for(
        self, handle: Any, spec: dict[str, Any], timeout_ms: int
    ) -> None:
        """Block until a condition holds (element/text/URL/title).

        Must be condition-based (driver events/polling), never a
        fixed sleep.  Raises BrowserException(TIMEOUT) when the
        condition does not hold within ``timeout_ms``.
        """
        self._unsupported("waiting for page conditions")

    def screenshot(self, handle: Any) -> bytes:
        """Capture the page as PNG bytes."""
        self._unsupported("screenshots")


class PlaywrightBackend(BrowserBackend):
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
            self._contexts[name] = self._browser.new_context(**kwargs)
        except Exception as e:
            raise BrowserException(
                f"Could not create browser profile {name!r}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"profile": name},
            ) from e
        return name

    def set_active_profile(self, name: str) -> None:
        context = self._contexts.get(name)
        if context is None:
            raise BrowserException(
                f"Unknown browser profile {name!r}",
                code=BrowserErrorCode.OPERATION_FAILED,
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
