"""
Test script to verify the rebuild endpoint works properly.
"""
import requests
import time

BASE_URL = "http://localhost:5000"

def test_rebuild():
    """Test the rebuild endpoint."""
    print("=" * 60)
    print("Testing /api/admin/rebuild endpoint")
    print("=" * 60)
    
    # Trigger rebuild
    print("\n[1] Triggering rebuild from cleaned_pages...")
    response = requests.post(f"{BASE_URL}/api/admin/rebuild")
    print(f"Response: {response.status_code}")
    print(f"Message: {response.json()}")
    
    # Wait for completion
    print("\n[2] Waiting for rebuild to complete...")
    for i in range(60):  # Wait up to 60 seconds
        time.sleep(2)
        status = requests.get(f"{BASE_URL}/api/admin/status").json()
        print(f"  Status: {status.get('message', status.get('status', 'unknown'))}")
        
        if status.get('status') == 'complete' or status.get('stage') == 'complete':
            print("\n[OK] Rebuild completed!")
            break
        elif status.get('status') == 'error':
            print(f"\n[ERROR] Rebuild failed: {status.get('error')}")
            return
    
    # Get stats
    print("\n[3] Getting index stats...")
    stats = requests.get(f"{BASE_URL}/api/admin/stats").json()
    print(f"  Has index: {stats.get('has_index')}")
    print(f"  Total chunks: {stats.get('total_chunks')}")
    print(f"  Pages scraped: {stats.get('pages_scraped')}")
    
    # Test query for establishment year
    print("\n[4] Testing RAG query: 'When was VNRVJIET established?'")
    response = requests.post(f"{BASE_URL}/api/chat", json={"query": "When was VNRVJIET established?"})
    result = response.json()
    print(f"\n  Answer: {result.get('answer', 'No answer')[:500]}")
    print(f"  Confidence: {result.get('confidence')}")
    print(f"  Grounded: {result.get('grounded')}")
    print(f"  Sources: {result.get('sources_count')}")
    
    # Check if 1995 is in the answer
    if "1995" in result.get('answer', ''):
        print("\n[OK] SUCCESS! Answer correctly contains '1995'")
    else:
        print("\n[FAIL] Answer does not contain '1995'")
        
    # Test another query
    print("\n[5] Testing RAG query: 'What programs does VNRVJIET offer?'")
    response = requests.post(f"{BASE_URL}/api/chat", json={"query": "What programs does VNRVJIET offer?"})
    result = response.json()
    print(f"\n  Answer: {result.get('answer', 'No answer')[:500]}")
    print(f"  Confidence: {result.get('confidence')}")

if __name__ == "__main__":
    test_rebuild()
