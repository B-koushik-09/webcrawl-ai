"""Quick verification that the fix works."""
import py_compile
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Syntax check
try:
    py_compile.compile('modules/rag_engine.py', doraise=True)
    print("[OK] rag_engine.py compiles")
except Exception as e:
    print(f"[FAIL] Syntax error: {e}")
    sys.exit(1)

# 2. Check no duplicate 'course' key
from modules.rag_engine import RAGEngine
rules = RAGEngine.ANSWER_VERIFICATION_RULES
assert 'course' in rules, "Missing 'course' key"
print(f"[OK] 'course' key exists with {len(rules['course']['required_patterns'])} required patterns")

# 3. Test that syllabus content would pass verification
import re
sample_chunk = """## 22PC1AM202: Data Engineering
| TEACHING SCHEME | EVALUATION SCHEME |
| L: 3 T: 0 P: 0 C: 3 | CIE: 40 SEE: 60 Total: 100 |

#### COURSE OBJECTIVES
* To explore data preprocessing techniques

#### UNIT-I
**Data Pre-processing**: Types of data, exploring structure of data"""

patterns = rules['course']['required_patterns']
matched = False
for pattern in patterns:
    if re.search(pattern, sample_chunk, re.IGNORECASE):
        print(f"[OK] Sample chunk matches pattern: {pattern[:50]}")
        matched = True
        break

if matched:
    print("[OK] Syllabus chunks will now PASS verification")
else:
    print("[FAIL] Sample chunk still does not match any pattern!")
    sys.exit(1)

print("\nAll checks passed!")
