#!/usr/bin/env python3
"""Emit stable public Gooaye Telegram message data for cron monitoring."""
from __future__ import annotations

import errno
import json
import random
import re
import socket
import sys
import time
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gooaye_episode_state import observe_messages

URL = "https://t.me/s/Gooaye"
# ponytail: fixed 3-attempt window; add a circuit breaker only if tunnel outages need suppression.
MAX_FETCH_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 4.0
BACKOFF_JITTER_SECONDS = 0.5

_NETWORK_UNREACHABLE_ERRNOS = frozenset({errno.EHOSTUNREACH, errno.ENETUNREACH})
_NETWORK_TRANSIENT_ERRNOS = frozenset({
    errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET, errno.ETIMEDOUT,
})


def _error_kind(exc):
    seen = set()
    pending = [exc]
    messages = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(str(current).lower())
        if isinstance(current, socket.gaierror):
            return "network_dns"
        if isinstance(current, HTTPError):
            status = current.code or 0
            if status in {408, 425, 429} or status >= 500:
                return "network_http"
            return "source_error"
        if isinstance(current, OSError):
            if current.errno in _NETWORK_UNREACHABLE_ERRNOS:
                return "network_unreachable"
            if current.errno in _NETWORK_TRANSIENT_ERRNOS:
                return "network_transient"
        for cause in (getattr(current, "reason", None), current.__cause__, current.__context__):
            if cause is not None:
                pending.append(cause)
    message = " ".join(messages)
    if "no route to host" in message or "network is unreachable" in message:
        return "network_unreachable"
    if (
        "temporary failure in name resolution" in message
        or "name or service not known" in message
        or "nodename nor servname provided" in message
    ):
        return "network_dns"
    if "timed out" in message or "timeout" in message:
        return "network_timeout"
    if "connection reset" in message or "connection aborted" in message or "connection refused" in message:
        return "network_transient"
    return "source_error"


class MonitorSourceError(Exception):
    def __init__(self, cause, attempts):
        self.cause = cause
        self.attempts = attempts
        self.error_kind = _error_kind(cause)
        self.retryable = self.error_kind.startswith("network_")
        super().__init__(str(cause))


def _fetch_html():
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            # A new request/connection per attempt avoids reusing a stale tunnel/socket.
            request = Request(URL, headers={
                "Connection": "close",
                "User-Agent": "Hermes-Gooaye-Monitor/1.0",
            })
            with urlopen(request, timeout=20) as response:
                return response.read()
        except Exception as exc:
            error_kind = _error_kind(exc)
            if not error_kind.startswith("network_") or attempt == MAX_FETCH_ATTEMPTS:
                raise MonitorSourceError(exc, attempt) from exc
            base_delay = min(
                MAX_BACKOFF_SECONDS,
                INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            )
            time.sleep(base_delay + random.uniform(0, BACKOFF_JITTER_SECONDS))


def _error_summary(cause, error_kind):
    detail = " ".join(str(cause).split())
    if len(detail) > 180:
        detail = detail[:177] + "..."
    return f"{error_kind}: {detail}" if detail else error_kind


class ChannelParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.messages = []
        self.current = None
        self.in_text = False
        self.text_parts = []
        self.in_time = False
        self.time_value = None
        self.in_post_link = False
        self.post_link = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "div" and "tgme_widget_message" in classes:
            self.current = {"id": None, "datetime": None, "text": ""}
            self.text_parts = []
            self.time_value = None
            self.post_link = None
        elif self.current is not None and tag == "div" and "tgme_widget_message_text" in classes:
            self.in_text = True
        elif self.current is not None and tag == "time":
            self.in_time = True
            self.time_value = attrs.get("datetime")
        elif self.current is not None and tag == "a":
            href = attrs.get("href", "")
            if re.fullmatch(r"https://t\.me/Gooaye/\d+", href):
                self.post_link = href

    def handle_data(self, data):
        if self.current is not None and self.in_text:
            self.text_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "time" and self.current is not None:
            self.in_time = False
        elif tag == "div" and self.current is not None and self.in_text:
            self.in_text = False
        elif tag == "div" and self.current is not None:
            if self.post_link:
                self.current["id"] = self.post_link.rsplit("/", 1)[-1]
                self.current["datetime"] = self.time_value
                self.current["text"] = " ".join("".join(self.text_parts).split())
                self.messages.append(self.current)
            self.current = None


def extract_messages(html):
    starts = list(re.finditer(r'data-post="Gooaye/(\d+)"', html))
    messages = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(html)
        block = html[match.start():end]
        time_match = re.search(r'<time[^>]+datetime="([^"]+)"', block)
        text_match = re.search(r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>', block, re.S)
        if not time_match:
            continue
        text = ""
        if text_match:
            cleaner = HTMLParser()
            parts = []
            class TextParser(HTMLParser):
                def handle_data(self, data):
                    parts.append(data)
            parser = TextParser(convert_charrefs=True)
            parser.feed(text_match.group(1))
            text = " ".join("".join(parts).split())
        messages.append({"id": match.group(1), "datetime": time_match.group(1), "text": text})
    return messages[-100:]


def main():
    html = _fetch_html()
    messages = extract_messages(html.decode("utf-8", errors="replace"))
    observe_messages(messages)
    latest = messages[-1] if messages else None
    payload = {
        "channel": "Gooaye",
        "url": URL,
        "source_status": "SOURCE_CHECKED",
        "message_count": len(messages),
        "contentful_message_count": sum(bool(message.get("text")) for message in messages),
        "latest_post_id": latest.get("id") if latest else None,
        "latest_post_datetime": latest.get("datetime") if latest else None,
        "messages": messages,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        source_error = exc.cause if isinstance(exc, MonitorSourceError) else exc
        error_kind = exc.error_kind if isinstance(exc, MonitorSourceError) else _error_kind(exc)
        retryable = exc.retryable if isinstance(exc, MonitorSourceError) else False
        print(json.dumps({
            "channel": "Gooaye",
            "error": _error_summary(source_error, error_kind),
            "error_kind": error_kind,
            "next_action": "retry on next scheduled run" if retryable else "inspect source URL/response",
            "attempts": exc.attempts if isinstance(exc, MonitorSourceError) else 1,
            "retryable": retryable,
            "source_status": "SOURCE_FAILED",
            "url": URL,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        sys.exit(1)
