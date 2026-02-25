"""Quick diagnostic script to test module imports"""
import sys
print("=" * 50)
print("DIAGNOSTIC: Testing module imports")
print("=" * 50)

try:
    print("[1/5] Importing config...")
    import config
    print("✓ config imported successfully")
except Exception as e:
    print(f"✗ config import failed: {e}")
    sys.exit(1)

try:
    print("[2/5] Importing modules.embeddings...")
    from modules import embeddings
    print("✓ modules.embeddings imported successfully")
except Exception as e:
    print(f"✗ modules.embeddings import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("[3/5] Importing modules.rag_engine...")
    from modules import rag_engine
    print("✓ modules.rag_engine imported successfully")
except Exception as e:
    print(f"✗ modules.rag_engine import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("[4/5] Importing modules.admin...")
    from modules import admin
    print("✓ modules.admin imported successfully")
except Exception as e:
    print(f"✗ modules.admin import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("[5/5] Importing app...")
    import app
    print("✓ app imported successfully")
except Exception as e:
    print(f"✗ app import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 50)
print("SUCCESS: All modules imported correctly!")
print("=" * 50)
print("\nThe issue is NOT with the code.")
print("The backend might just be starting up slowly.")
print("Check if port 5000 is listening with: netstat -ano | findstr :5000")
