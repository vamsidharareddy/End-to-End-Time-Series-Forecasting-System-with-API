"""
Test script for API endpoints after training completes.
Usage: python test_api.py
"""
import time
import sys

try:
    import requests
except ImportError:
    print("requests not installed. Install with: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8000"
TIMEOUT = 5


def test_health():
    """Test /health endpoint"""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        print("✓ /health - OK")
        return True
    except Exception as e:
        print(f"✗ /health - {e}")
        return False


def test_states():
    """Test /states endpoint"""
    try:
        resp = requests.get(f"{BASE_URL}/states", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        states = data.get("states", [])
        assert len(states) > 0, "No states returned"
        print(f"✓ /states - {len(states)} states found: {states[:3]}...")
        return True
    except Exception as e:
        print(f"✗ /states - {e}")
        return False


def test_forecast_single():
    """Test /forecast/{state} endpoint"""
    try:
        resp = requests.get(f"{BASE_URL}/forecast/California", timeout=TIMEOUT)
        assert resp.status_code == 200, f"Forecast failed: {resp.status_code}"
        data = resp.json()
        assert "state" in data
        assert "best_model" in data
        assert "forecast" in data
        print(f"✓ /forecast/California - Model: {data['best_model']}, Forecast points: {len(data['forecast'])}")
        return True
    except Exception as e:
        print(f"✗ /forecast/California - {e}")
        return False


def test_forecast_custom_horizon():
    """Test /forecast/{state} with custom horizon"""
    try:
        resp = requests.get(f"{BASE_URL}/forecast/Texas?horizon=12", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["forecast"]) == 12, f"Expected 12 forecasts, got {len(data['forecast'])}"
        print(f"✓ /forecast/Texas?horizon=12 - {len(data['forecast'])} week forecast")
        return True
    except Exception as e:
        print(f"✗ /forecast/Texas?horizon=12 - {e}")
        return False


def test_summary():
    """Test /summary endpoint"""
    try:
        resp = requests.get(f"{BASE_URL}/summary", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        rows = data.get("rows", [])
        print(f"✓ /summary - {len(rows)} states evaluated")
        if rows:
            print(f"  Sample: {rows[0]}")
        return True
    except Exception as e:
        print(f"✗ /summary - {e}")
        return False


def test_bulk():
    """Test /forecast/bulk endpoint"""
    try:
        resp = requests.post(f"{BASE_URL}/forecast/bulk?horizon=8", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        results = data.get("results", [])
        print(f"✓ /forecast/bulk - {len(results)} forecasts generated")
        return True
    except Exception as e:
        print(f"✗ /forecast/bulk - {e}")
        return False


def main():
    print("=" * 60)
    print("Testing Forecasting API")
    print("=" * 60)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            health = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            if health.status_code == 200:
                break
        except:
            if attempt < max_retries - 1:
                print(f"API not ready. Waiting... (attempt {attempt+1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"✗ Could not connect to API at {BASE_URL}")
                print("  Ensure API is running: uvicorn api:app --reload")
                return
    
    tests = [
        test_health,
        test_states,
        test_forecast_single,
        test_forecast_custom_horizon,
        test_summary,
        test_bulk,
    ]
    
    results = [test() for test in tests]
    
    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
