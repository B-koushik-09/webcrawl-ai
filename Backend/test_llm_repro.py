
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from modules.llm_generator import LLMGenerator

def test_repro():
    gen = LLMGenerator()
    if not gen.is_available():
        print("LLM not available. Set HF_TOKEN.")
        return

    query = "Does VNRVJIET have industry-related curriculum features?"
    context_chunks = [
        {
            "source_name": "VNRVJIET Contact Information (webpage)",
            "content": "## Does VNRVJIET have industry-related curriculum features?\nThe curriculum emphasizes relevant skills, project-based learning, and industry readiness."
        }
    ]

    print("\n" + "="*50)
    print(f"Testing Query: {query}")
    print("="*50)
    
    res = gen.generate_answer(query, context_chunks)
    
    print(f"\nLLM ANSWER:\n{res.answer}")
    print(f"\nGeneration Time: {res.generation_time:.2f}s")
    print(f"Model: {res.model}")

if __name__ == "__main__":
    test_repro()
