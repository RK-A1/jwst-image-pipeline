"""
Flickr REST client for the NASA Webb account.

Deliberately small: the four read-only methods the ingest needs, with request
timeouts, bounded retries on transient failures, and pacing well under Flickr's
3,600 requests an hour. Flickr reports most errors as HTTP 200 with `stat: fail`, so
both the HTTP status and the body are checked.

The API key travels as a query parameter, so requests' own exception messages contain
it. Every error raised from here is rebuilt with the key redacted and without the
original exception chained, because Airflow writes full tracebacks into task logs.
"""

import logging
import time
from collections.abc import Callable

import requests

from include.jwst_pipeline.config import FLICKR_USER

log = logging.getLogger(__name__)

API_URL = "https://api.flickr.com/services/rest/"
PER_PAGE = 500

# Largest first. "Original" is not always offered, so fall back through the rest.
SIZE_PREFERENCE = [
    "Original", "Large 2048", "Large 1600", "Large", "Medium 800", "Medium 640", "Medium",
]

# Flickr's "service currently unavailable". Every other stat=fail code is permanent for
# the request that produced it (bad photo id, invalid key, and so on).
TRANSIENT_CODES = {105}


class FlickrError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, transient: bool = False):
        super().__init__(message)
        self.code = code
        self.transient = transient


class FlickrClient:
    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        pause_s: float = 0.3,
        max_attempts: int = 4,
        backoff_s: float = 2.0,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ValueError("a Flickr API key is required")
        self._key = api_key
        self._http = session or requests.Session()
        self._pause_s = pause_s
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._timeout_s = timeout_s
        self._sleep = sleep

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***")

    def call(self, method: str, **params) -> dict:
        query = {
            "method": method, "api_key": self._key,
            "format": "json", "nojsoncallback": 1, **params,
        }
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._http.get(API_URL, params=query, timeout=self._timeout_s)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise FlickrError(f"{method}: HTTP {resp.status_code}", transient=True)
                if resp.status_code >= 400:
                    raise FlickrError(f"{method}: HTTP {resp.status_code}")
                try:
                    body = resp.json()
                except ValueError:
                    # An outage page served as HTML rather than JSON.
                    raise FlickrError(f"{method}: response was not JSON", transient=True) from None
                if body.get("stat") != "ok":
                    code = body.get("code")
                    raise FlickrError(
                        f"{method}: {body.get('message', 'unknown error')} (code {code})",
                        code=code, transient=code in TRANSIENT_CODES,
                    )
                self._sleep(self._pause_s)
                return body
            except FlickrError as exc:
                error = exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                error = FlickrError(
                    f"{method}: {type(exc).__name__}: {self._redact(str(exc))}", transient=True
                )
            except requests.RequestException as exc:
                error = FlickrError(f"{method}: {type(exc).__name__}: {self._redact(str(exc))}")

            if not error.transient or attempt == self._max_attempts:
                raise error from None
            delay = self._backoff_s * 2 ** (attempt - 1)
            log.warning("%s (attempt %d/%d), retrying in %.0fs",
                        error, attempt, self._max_attempts, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")

    def resolve_user_id(self, username: str = FLICKR_USER) -> str:
        # urls.lookupUser takes the path alias; people.findByUsername does not find it.
        body = self.call("flickr.urls.lookupUser", url=f"https://www.flickr.com/photos/{username}/")
        return body["user"]["id"]

    def list_photo_ids(self, user_id: str) -> list[str]:
        """Every public photo id, newest first, de-duplicated."""
        ids: list[str] = []
        page, total = 1, 0
        while True:
            photos = self.call(
                "flickr.people.getPublicPhotos", user_id=user_id, per_page=PER_PAGE, page=page
            )["photos"]
            total = int(photos["total"])
            ids.extend(p["id"] for p in photos["photo"])
            log.info("listed page %d/%s (%d ids so far)", page, photos["pages"], len(ids))
            if page >= int(photos["pages"]):
                break
            page += 1

        # An upload during pagination shifts every later page by one, repeating an id.
        unique = list(dict.fromkeys(ids))
        if len(unique) != total:
            log.warning("Flickr reported %d photos but listing returned %d; "
                        "anything missed is picked up on the next run", total, len(unique))
        return unique

    def photo_metadata(self, photo_id: str) -> dict:
        return parse_photo_info(self.call("flickr.photos.getInfo", photo_id=photo_id)["photo"])

    def best_image_url(self, photo_id: str) -> str:
        return pick_size(self.call("flickr.photos.getSizes", photo_id=photo_id)["sizes"]["size"])


def parse_photo_info(info: dict) -> dict:
    """Reduce a photos.getInfo payload to the columns the warehouse stores."""
    return {
        "photo_id": info["id"],
        "title": info["title"]["_content"],
        "description": info["description"]["_content"],
        # Flickr returns tags with spaces removed: "star cluster" arrives as "starcluster".
        "tags": [t["_content"] for t in info.get("tags", {}).get("tag", [])],
        "date_taken": info.get("dates", {}).get("taken") or None,  # "YYYY-MM-DD HH:MM:SS"
    }


def pick_size(sizes: list[dict]) -> str:
    """URL of the largest preferred rendition, or the last listed if none match."""
    if not sizes:
        raise FlickrError("photo has no downloadable sizes")
    by_label = {s["label"]: s["source"] for s in sizes}
    for label in SIZE_PREFERENCE:
        if label in by_label:
            return by_label[label]
    return sizes[-1]["source"]
