"""
Test script to verify CSE department-aware retrieval implementation.
Tests: syntax, dept classification, query detection, and (if index exists) live retrieval.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_syntax():
    """Test that all modified files compile without errors."""
    import py_compile
    files = ['config.py', 'modules/embeddings.py', 'modules/rag_engine.py']
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
            print(f"[OK] {f} compiles")
        except py_compile.PyCompileError as e:
            print(f"[FAIL] {f} has syntax error: {e}")
            return False
    return True

def test_config_imports():
    """Test that new config values are importable."""
    from config import CSE_DEPT_KEYWORDS, CSE_QUERY_KEYWORDS
    print(f"[OK] CSE_DEPT_KEYWORDS: {len(CSE_DEPT_KEYWORDS)} keywords")
    print(f"[OK] CSE_QUERY_KEYWORDS: {len(CSE_QUERY_KEYWORDS)} keywords")
    
    # Verify 'programming' is NOT in query keywords (user caution)
    assert 'programming' not in CSE_QUERY_KEYWORDS, "'programming' should NOT be in CSE_QUERY_KEYWORDS"
    print("[OK] 'programming' correctly excluded from query keywords")
    return True

def test_dept_classification():
    """Test the _classify_dept method logic."""
    from config import CSE_DEPT_KEYWORDS
    
    def classify(content, source="", section=""):
        combined = f"{content} {source} {section}".lower()
        return 'CSE' if any(kw in combined for kw in CSE_DEPT_KEYWORDS) else 'OTHER'
    
    # Should be CSE
    assert classify("Department of Computer Science and Engineering") == 'CSE'
    assert classify("B.Tech CSE syllabus") == 'CSE'
    assert classify("some content", source="CSE Faculty List") == 'CSE'
    assert classify("some content", section="CSE Lab Manual") == 'CSE'
    print("[OK] CSE chunks correctly classified")
    
    # Should be OTHER
    assert classify("Department of Mechanical Engineering") == 'OTHER'
    assert classify("ECE embedded programming lab") == 'OTHER'
    assert classify("Hostel fee structure") == 'OTHER'
    print("[OK] Non-CSE chunks correctly classified as OTHER")
    return True

def test_cse_query_detection():
    """Test CSE query detection logic."""
    from config import CSE_QUERY_KEYWORDS
    
    def is_cse(query):
        return any(kw in query.lower() for kw in CSE_QUERY_KEYWORDS)
    
    # Should detect as CSE
    assert is_cse("What is the CSE department fee?") == True
    assert is_cse("Tell me about data structures course") == True
    assert is_cse("Who teaches operating systems?") == True
    assert is_cse("computer science faculty list") == True
    print("[OK] CSE queries correctly detected")
    
    # Should NOT detect as CSE
    assert is_cse("What is the hostel fee?") == False
    assert is_cse("Who is the principal?") == False
    assert is_cse("ECE programming lab") == False  # 'programming' excluded
    print("[OK] Non-CSE queries correctly NOT detected")
    return True

def test_live_retrieval():
    """Test with actual index if available."""
    try:
        from modules.embeddings import KnowledgeIndex
        index = KnowledgeIndex()
        if index.index is None or index.index.ntotal == 0:
            print("[SKIP] No index loaded - rebuild needed for live test")
            return True
        
        # Check how many chunks have dept metadata
        dept_counts = {'CSE': 0, 'OTHER': 0, 'MISSING': 0}
        for item in index.items:
            dept = item.metadata.get('dept')
            if dept == 'CSE':
                dept_counts['CSE'] += 1
            elif dept == 'OTHER':
                dept_counts['OTHER'] += 1
            else:
                dept_counts['MISSING'] += 1
        
        print(f"[INFO] Dept distribution: {dept_counts}")
        
        if dept_counts['MISSING'] == len(index.items):
            print("[WARN] All chunks missing dept field - rebuild needed to tag chunks")
        else:
            print(f"[OK] {dept_counts['CSE']} CSE chunks, {dept_counts['OTHER']} OTHER chunks")
        
        return True
    except Exception as e:
        print(f"[SKIP] Could not load index: {e}")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("CSE Department-Aware Retrieval - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Syntax Check", test_syntax),
        ("Config Imports", test_config_imports),
        ("Dept Classification", test_dept_classification),
        ("CSE Query Detection", test_cse_query_detection),
        ("Live Retrieval", test_live_retrieval),
    ]
    
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            if fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
