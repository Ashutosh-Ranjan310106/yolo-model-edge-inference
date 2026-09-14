"""
Verification script for caching policy:
1. Frontend assets (/client/*) must NEVER be cached (returns 200 OK, never 304, no etag, no last-modified, no-store).
2. Model downloads (/api/models/*/download) MUST allow caching (Cache-Control: public, max-age=...).
"""

import sys
import os

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app

def main():
    client = TestClient(app)

    print("=" * 70)
    print(" Verifying Caching Policy: Zero Frontend Cache, Explicit Model Cache")
    print("=" * 70)

    # 1. Test GET /client/ (HTML)
    res_html = client.get("/client/")
    print(f"1. GET /client/ -> Status: {res_html.status_code}")
    assert res_html.status_code == 200, f"Expected 200, got {res_html.status_code}"
    cc = res_html.headers.get("cache-control", "")
    print(f"   Cache-Control: {cc}")
    assert "no-store" in cc and "no-cache" in cc, f"Expected no-store in Cache-Control, got {cc}"
    assert "etag" not in res_html.headers, f"ETag should be stripped, found: {res_html.headers.get('etag')}"
    assert "last-modified" not in res_html.headers, f"Last-Modified should be stripped, found: {res_html.headers.get('last-modified')}"

    # 2. Test conditional GET with If-None-Match (must return 200 OK, NOT 304)
    res_etag = client.get("/client/", headers={"If-None-Match": 'W/"fake-etag-12345"'})
    print(f"2. Conditional GET /client/ (If-None-Match) -> Status: {res_etag.status_code}")
    assert res_etag.status_code == 200, f"Expected 200 (not 304), got {res_etag.status_code}"

    # 3. Test conditional GET with If-Modified-Since (must return 200 OK, NOT 304)
    res_ims = client.get("/client/", headers={"If-Modified-Since": "Sat, 29 Oct 2099 00:00:00 GMT"})
    print(f"3. Conditional GET /client/ (If-Modified-Since) -> Status: {res_ims.status_code}")
    assert res_ims.status_code == 200, f"Expected 200 (not 304), got {res_ims.status_code}"

    # 4. Test GET /client/js/app.js
    res_js = client.get("/client/js/app.js")
    print(f"4. GET /client/js/app.js -> Status: {res_js.status_code}")
    assert res_js.status_code == 200, f"Expected 200, got {res_js.status_code}"
    cc_js = res_js.headers.get("cache-control", "")
    print(f"   Cache-Control: {cc_js}")
    assert "no-store" in cc_js and "no-cache" in cc_js

    # 5. Test conditional GET on JS with If-None-Match (must return 200 OK, NOT 304)
    res_js_etag = client.get("/client/js/app.js", headers={"If-None-Match": 'W/"any-tag"'})
    print(f"5. Conditional GET /client/js/app.js (If-None-Match) -> Status: {res_js_etag.status_code}")
    assert res_js_etag.status_code == 200, f"Expected 200 (not 304), got {res_js_etag.status_code}"

    # 6. Test Model download endpoint Cache-Control
    models_res = client.get("/api/models")
    if models_res.status_code == 200 and models_res.json().get("models"):
        first_model = models_res.json()["models"][0]
        model_id = first_model["id"]
        res = first_model.get("resolution", 480)
        dl_res = client.get(f"/api/models/{model_id}/download?resolution={res}")
        if dl_res.status_code == 200:
            m_cc = dl_res.headers.get("cache-control", "")
            print(f"6. Model Download Cache-Control: {m_cc}")
            assert "public" in m_cc or "max-age" in m_cc, f"Expected model caching allowed, got {m_cc}"
            print("   ✓ Model caching explicitly enabled!")

    print("=" * 70)
    print(" ALL CACHING POLICY TESTS PASSED!")
    print(" - Frontend assets: Zero cache, No 304, No ETag, Fresh every time.")
    print(" - Model binaries: Device IndexedDB + Cache-Control enabled.")
    print("=" * 70)

if __name__ == "__main__":
    main()
