"""
Quick test script for the new Google GenAI LLM generator.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from modules.llm_generator import LLMGenerator

def test_llm():
    print("=" * 60)
    print("Testing Google GenAI LLM Generator")
    print("=" * 60)
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set in .env file")
        return
    
    print(f"[OK] API Key found: ...{api_key[-4:]}")
    
    # Create generator
    gen = LLMGenerator()
    
    if not gen.is_available():
        print("[ERROR] LLM Generator not available")
        return
    
    print(f"[OK] LLM Generator initialized with model: {gen.model_name}")
    
    # Test query
    test_context = [
        {
            "content": "VNR VJIET offers 14 B.Tech programs and 13 M.Tech programs. The college was established in 1995.",
            "source_name": "About VNRVJIET"
        },
        {
            "content": "The tuition fee for B.Tech is Rs. 1,30,000 per year.",
            "source_name": "Fee Structure" 
        }
    ]
    
    queries = [
        "What is the fee for B.Tech?",
        "How many programs does the college offer?",
        "When was the college established?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            result = gen.generate_answer(query, test_context)
            
            # Use utf-8 encoding to handle special characters
            answer_text = result.answer.encode('utf-8', errors='replace').decode('utf-8')
            
            print(f"Answer: {answer_text}")
            print(f"Model: {result.model}")
            print(f"Provider: {result.provider}")
            print(f"Time: {result.generation_time:.2f}s")
            print(f"Citations: {', '.join(result.citations_used)}")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_llm()
