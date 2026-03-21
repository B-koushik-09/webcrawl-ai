"""
CollegeWeb AI - RAG (Retrieval-Augmented Generation) Engine
Generates accurate, grounded answers using retrieved context.
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAG_CONFIG, CSE_QUERY_KEYWORDS, GENERAL_KEYWORDS, OTHER_BRANCH_KEYWORDS, CSE_SUB_DEPT_KEYWORDS
from modules.embeddings import KnowledgeIndex, KnowledgeItem, get_knowledge_index


@dataclass
class Citation:
    """Represents a source citation."""
    source_type: str  # 'webpage' or 'pdf'
    source_name: str
    source_url: str
    page_number: Optional[int]
    relevance_score: float
    snippet: str

@dataclass
class RAGResponse:
    """Represents a RAG-generated response."""
    query: str
    answer: str
    citations: List[Citation]
    confidence: float
    grounded: bool  # True if answer is based on retrieved content
    raw_context: str


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for generating grounded answers.
    Uses retrieved knowledge to create accurate responses.
    """
    
    # Response templates
    NO_INFO_RESPONSE = "I couldn't find any official information about this in the college website or documents. Please contact the college administration for accurate information."
    
    LOW_CONFIDENCE_PREFIX = "Based on limited information available: "
    
    # Question type patterns for better response formatting
    QUESTION_PATTERNS = {
        'fee': [r'\bfee\b', r'\btuition\b', r'\bcost\b', r'\bpayment\b', r'\bscholarship\b', r'\bexpense'],
        'admission': [r'\badmission\b', r'\beligib', r'\brequirement', r'\bapply\b', r'\benroll', r'\bentrance'],
        'exam': [r'\bexam\b', r'\btest\b', r'\bassessment\b', r'\bgrade\b', r'\bmark\b', r'\bresult', r'\bpass'],
        'schedule': [r'\bdate\b', r'\bwhen\b', r'\bschedule\b', r'\bcalendar\b', r'\btiming\b', r'\bdeadline'],
        'contact': [r'\bcontact\b', r'\bphone\b', r'\bemail\b', r'\baddress\b', r'\blocation', r'\bwhere\s+(?:is|are)', r'\blocated\b'],
        'course': [r'\bcourse\b', r'\bsyllabus\b', r'\bsubject\b', r'\bcurriculum\b', r'\bprogram', r'\bbranch', r'\boffers\b', r'\bintake\b'],
        'placement': [r'\bplacement\b', r'\bjob\b', r'\bcareer\b', r'\brecruit\b', r'\bintern', r'\bpackage', r'\bsalary'],
        'principal': [r'\bprincipal\b', r'\bdirector\b', r'\bdean\b', r'\bhod\b', r'\bhead\b', r'\bchairman'],
        'faculty': [r'\bdesignation\b', r'\bprofessor\b', r'\bfaculty\b', r'\bteacher\b', r'\blecturer\b', r'\bspeciali[sz]ation\b', r'\bqualification\b', r'\bexperience\s+of\b', r'\bdepartment\s+of\b'],
        'rules': [r'\brule\b', r'\bpolicy\b', r'\bguideline\b', r'\bregulation\b', r'\bconduct', r'\bdiscipline'],
        'facilities': [r'\bfacilit', r'\blab\b', r'\blibrary\b', r'\bsport', r'\bgym\b', r'\bwifi\b', r'\binfrastructure'],
        'hostel': [r'\bhostel\b', r'\baccommodation\b', r'\bmess\b', r'\broom\b', r'\bdormitory', r'\bresidence'],
        'about': [
            r'\babout\s+(?:the\s+)?(?:college|vnr|institute|university|campus)', r'\bhistory\b', r'\bestablish', r'\bfound', r'\bvision\b', r'\bmission\b',
            r'\bfull\s*name\b', r'\bstands\s*for\b', r'\bmean(?:ing|s)\b', r'\bwhat\s*is\s*vnr',
        ],
    }
    
    # Answer-Type Verification Rules - chunks MUST contain these patterns to be valid
    ANSWER_VERIFICATION_RULES = {
        'fee': {
            'required_patterns': [
                r'₹\s*[\d,]+',           # ₹1,30,000 format
                r'Rs\.?\s*[\d,]+',        # Rs. 1,30,000 or Rs 130000
                r'INR\s*[\d,]+',          # INR 130000
                r'\d+[,\d]*\s*(?:per\s+(?:year|annum|semester|month))',  # 130000 per year
                r'tuition\s*(?:fee)?\s*(?:is|:)?\s*[\d₹]',  # tuition fee is/: followed by number
                r'(?:annual|yearly|semester)\s*fee\s*(?:is|:)?\s*[\d₹]',  # annual fee is
                r'(?:payment|fee)\s+guidelines',  # "Payment Guidelines"
                r'(?:how\s+to\s+pay|payment\s+process)',
                r'(?:online|digital)\s+(?:fee|payment)',
                r'(?:transaction|upi|neft|rtgs|demand\s+draft)',
                r'(?:bank|account)\s+details',
            ],
            'not_found_message': "I couldn't find official fee structure information with actual amounts. The fee structure PDF may not be available on the website. Please contact the college administration or visit the official website's Admissions section."
        },

        'admission': {
            'required_patterns': [
                r'eligib(?:le|ility)',
                r'(?:10\+2|12th|intermediate)',
                r'(?:minimum|required)\s*(?:marks?|percentage|score)',
                r'(?:entrance|qualifying)\s*(?:exam|test)',
                r'(?:apply|application)\s*(?:online|form|process)',
                r'(?:seat|intake)\s*(?:capacity|available)',
                r'EAMCET', r'ECET', r'JEE', r'GATE', r'PGECET',  # Specific exams
            ],
            'boost_patterns': [
                r'EAMCET', r'ECET', r'JEE', r'Main', r'Advanced',  # Boost B.Tech exams
                r'convener\s+quota', r'management\s+quota', r'category\s+b',
            ],
            'not_found_message': "I couldn't find detailed admission eligibility criteria. Please check the official Admissions page or contact the Admissions Office."
        },
        'exam': {
            'required_patterns': [
                r'(?:exam|test)\s*(?:date|schedule|pattern)',
                r'(?:mid|end|final)\s*(?:term|semester)\s*exam',
                r'(?:marks?|grade)\s*(?:distribution|weightage)',
                r'(?:pass(?:ing)?|minimum)\s*(?:marks?|percentage|criteria)',
                r'(?:internal|external)\s*(?:assessment|evaluation)',
                r'(?:question|paper)\s*(?:pattern|format)',
            ],
            'not_found_message': "I couldn't find detailed examination information. Please check the Academics section or contact the Examination Cell."
        },
        'schedule': {
            'required_patterns': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date formats: 01/02/2024
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}',  # January 15
                r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',  # 15 January
                r'(?:deadline|last\s+date|start\s+date|end\s+date)',
                r'(?:semester|session|academic\s+year)\s*(?:starts?|begins?|ends?)',
                r'\d{4}\s*-\s*\d{2,4}',  # Academic year: 2024-25
            ],
            'not_found_message': "I couldn't find specific dates or schedule information. Please check the college calendar or contact the administration."
        },
        'contact': {
            'required_patterns': [
                r'[\d\-\+\(\)\s]{10,}',  # Phone numbers
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
                r'(?:phone|mobile|tel|fax)\s*:?\s*[\d\-\+]',
                r'(?:email|mail)\s*:?\s*[a-zA-Z0-9._%+-]+@',
                r'(?:address|located|campus)\s*:?',
                r'(?:Bachupally|Nizampet|Hyderabad|Telangana|Medchal)',  # Specific locations
                r'(?:at|in)\s+(?:Bachupally|Nizampet)',
            ],
            'not_found_message': "I couldn't find specific contact details. Please visit the Contact Us page on the college website."
        },
        'course': {
            'required_patterns': [
                # Syllabus content patterns (unit descriptions, course structure)
                r'UNIT[\s-]*(?:I|II|III|IV|V|1|2|3|4|5)',
                r'COURSE\s+(?:OBJECTIVES|OUTCOMES)',
                r'\d{2}[A-Z]{2}\d[A-Z]{2}\d{3}',  # Course codes like 22PC1AM202
                r'(?:Teaching|TEACHING)\s+(?:Scheme|SCHEME)',
                r'(?:Evaluation|EVALUATION)\s+(?:Scheme|SCHEME)',
                r'L:\s*\d+.*(?:T|P|C):\s*\d+',  # L: 3 T: 0 P: 0 C: 3
                r'(?:TEXT\s+BOOKS?|REFERENCES)',
                r'CO-\d+',  # Course Outcome markers like CO-1
                # Programme/course listing patterns
                r'(?:credit|hours?)\s*:?\s*\d+',
                r'(?:semester|year)\s*[:-]?\s*(?:I|II|III|IV|V|VI|VII|VIII|\d)',
                r'(?:core|elective|mandatory|optional)\s*(?:subject|course|paper)',
                r'(?:subject|paper)\s*(?:code|name)\s*:?',
                r'(?:b\.?tech|m\.?tech|mba|mca|u\.?g\.?|p\.?g\.?)\s*(?:in)?',
                r'(?:programmes?|courses?)\s*(?:offered|offering|on\s+offer)',
                r'(?:intake|seats)\s*(?:per\s+year|is|:)?\s*\d+',
                r'(?:branches|departments)\s+(?:available|offered)',
                r'offers\s+\d+\s+(?:B\.?Tech|M\.?Tech|Ph\.?D|programmes)',
                r'total\s+(?:of\s+)?\d+\s+programmes',
                r'(?:curriculum|syllabus|subjects)',
                r'(?:CIE|SEE|SE):\s*\d+',  # Evaluation markers like CIE: 40 SEE: 60
            ],
            'boost_patterns': [
                r'UNIT[\s-]*(?:I|II|III|IV|V)',
                r'COURSE\s+OBJECTIVES',
                r'total\s+(?:number\s+of\s+)?(?:programmes?|courses?|branches)',
                r'offers\s+\d+\s+B\.?Tech',
                r'offers\s+\d+\s+M\.?Tech',
                r'The\s+Institute\s+offers\s+\d+',
                r'17\s+B\.?Tech.*programmes',
                r'15\s+M\.?Tech.*programmes',
                r'5\s+Ph\.?D.*programmes',
            ],
            'exclude_patterns': [
                # Exclude markdown table rows with S.No that confuse LLM
                r'\|\s*\d+\s*\|\s*\d{4}\s*\|',  # | 1 | 2021 | pattern
                r'\|\s*S\.\s*No\s*\|',  # Table header
            ],
            'not_found_message': "I couldn't find detailed course/programme information. Please check the Academics section or contact the department."
        },
        'placement': {
            'required_patterns': [
                # Direct CTC/package mentions (e.g., "CTC: Rs. 10 LPA")
                r'(?:package|salary|ctc|lpa|stipend|offer).*?[\d₹rs].*?[\d,\.]+',
                # Table format: "CTC (in Rs)" column header (placement PDFs use this)
                r'CTC\s*\(in\s*Rs\)',
                # Standalone numbers that look like packages (e.g., "92. 00" for 92 LPA in table cells)
                r'\b[1-9]\d{1,2}\.\s*\d{2}\b',  # Matches: 92. 00, 54. 00, 10. 00, etc.
                # Company names (table rows have company name + CTC)
                r'(?:GOOGLE|MICROSOFT|AMAZON|ORACLE|TCS|INFOSYS|WIPRO|COGNIZANT|ACCENTURE|DELOITTE|CAPGEMINI|JPMC|MICRON|NCR|DBS)',
                # Placement count indicators
                r'(?:placed|recruited|selected|offers)\s*(?:students?)?[\s:]*\d+',
                r'(?:total|students|placements?)[\s:]*\d+',
                # Campus/placement drive context
                r'(?:campus|placement)\s*(?:drive|interview|recruitment)',
                # Package descriptors
                r'(?:highest|average|median|minimum)\s*(?:package|salary|ctc)',
                # Training & Placement department/cell context
                r'(?:training|T\s*&\s*P|T\s*and\s*P)\s*(?:placement|cell|department|office|officer)',
                r'(?:placement)\s*(?:cell|department|office|officer)',
                # Faculty in placement context
                r'(?:professor|faculty|designation|coordinator).*(?:placement|training)',
                r'(?:placement|training).*(?:professor|faculty|designation|coordinator)',
            ],
            'not_found_message': "I couldn't find detailed placement information. Please contact the Training & Placement Cell."
        },
        'principal': {
            'required_patterns': [
                r'(?:dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|sri|smt)\s+[A-Z][A-Za-z]+',  # Dr. Name or Sri Name, allow all caps
                r'(?:principal|director|dean|hod)\s*:?\s*(?:dr\.?|prof\.?)?',
                r'(?:head\s+(?:of\s+)?(?:department|institution))\s*:?',
                r'(?:professor\s+and\s+head)',
                r'(?:chairman|chairperson)\s*[,:]',  # Chairman, Nominated
                r'governing\s+council',
                r'nominated\s+by\s+society',
                r'\|\s*\**Dr\.\s*Vadlana\s+Baby\**\s*\|',  # Specific match for HOD of CSE row in markdown table
                # HOD-specific patterns for department faculty pages
                r'(?:associate|assistant)?\s*professor\s+and\s+head',
                r'(?:incharge|in-charge)\s+hod',
                r'(?:department|dept)\s*(?:of|:)\s*(?:ece|eee|cse|mech|civil|it|aiml|iot)',
            ],
            'boost_patterns': [
                r'governing\s+council',
                r'chairman.*(?:nominated|society)',
                r'member\s+secretary',
                r'principal.*member',
                r'B\.\s*Chennakesava\s*Rao',  # Boost current principal
                r'Sagar\s+Yeruva',             # HOD of CSE-IOT and CSE-AIML
                r'Vadlana\s+Baby',             # HOD of CSE
                r'Poonam\s+Upadhyay',          # HOD of EEE
                r'Padma\s+Sai',                # HOD of ECE
                r'D\.\s*Suresh\s+Babu',        # Chairman
            ],
            'exclude_patterns': [
                # Exclude BOS/committee chairmen from faculty PDFs
                r'BOS\s+Chairman',
                r'Board\s+of\s+Studies.*Chairman',
                r'(?:as|acted\s+as)\s+(?:a\s+)?(?:BOS|IQAC|committee)\s+(?:Chairman|Convener)',
                r'Chairman\s+(?:BOS|Board|IAC|IQAC|Disciplinary)',
                r'C_cube\s+Consultants',  # Guest lecturer named "Chairman" of unrelated org
                r'Jaitley\s+Chairman',  # Specific noise: guest lecturer
                
                # AGGRESSIVE FILTERS FOR OLD LEADERSHIP (C D Naidu)
                r'C\.?\s*D\.?\s*Naidu',
                r'Dr\.\s*C\.?\s*D\.?\s*Naidu',
                r'Naidu\s*,\s*Principal',
                r'C\s*D\s*Naidu',
                r'C\.D\.Naidu',
                r'\bNaidu\b(?!\s+(?:Road|Street|College|Building))', # Match Naidu unless it's part of an address/place
                r'former\s+principal',
                r'past\s+principal',
            ],
            'not_found_message': "I couldn't find specific information about the principal/director/chairman. Please check the About Us or Governance page on the college website."
        },
        'faculty': {
            'required_patterns': [
                r'(?:assistant|associate)?\s*(?:professor|prof\.)',
                r'(?:designation|department|experience|specialization|email|qualification)',
                r'(?:teaching|research)\s*:?\s*\d+',
                r'@vnrvjiet\.(?:in|ac\.in|com)',
                r'(?:B\.?Tech|M\.?Tech|Ph\.?D|M\.?Sc|B\.?E)',
            ],
            'exclude_patterns': [
                r'Designation:\s*of\s+the\s+Paper',  # Garbage from PDFs
                r'Designation:\s*,',  # Empty designation
                r'Designation:\s*objective',
            ],
            'boost_patterns': [
                r'\|\s*\d+\s*\|\s*[A-Z0-9]+\s*\|',  # Standard table row with ID
                r'\|\s*Assistant\s+Professor\s*\|',
                r'\|\s*Associate\s+Professor\s*\|',
                r'\|\s*Professor\s*\|',
            ],
            'not_found_message': "I couldn't find information about this faculty member. Please check the respective department page or the Faculty Directory on the college website."
        },
        'rules': {
            'required_patterns': [
                r'(?:students?)\s*(?:must|should|shall|are\s+required)',
                r'(?:prohibited|not\s+allowed|banned)',
                r'(?:attendance|leave)\s*(?:policy|requirement)',
                r'(?:dress\s+code|uniform)',
                r'(?:disciplinary|punishment|penalty)',
                r'(?:academic\s+integrity|plagiarism)',
            ],
            'not_found_message': "I couldn't find specific rules or guidelines. Please check the Student Handbook or contact the Student Affairs office."
        },
        'facilities': {
            'required_patterns': [
                r'(?:laboratory|lab)\s*(?:facility|equipment)',
                r'(?:library)\s*(?:has|with|contains)',
                r'(?:sports?|gym|playground|ground)',
                r'(?:wifi|internet|computer)\s*(?:facility|access)',
                r'(?:air\s*condition|ac|projector)',
                r'(?:canteen|cafeteria|food\s+court)',
            ],
            'not_found_message': "I couldn't find detailed facilities information. Please check the Infrastructure section or visit the campus."
        },
        'hostel': {
            'required_patterns': [
                r'(?:hostel|accommodation)\s*(?:fee|charges|facility)',
                r'(?:room)\s*(?:type|facility|sharing)',
                r'(?:mess|food)\s*(?:facility|charges)',
                r'(?:boys?|girls?|women|men)\s*(?:hostel|accommodation)',
                r'(?:warden|supervisor)',
            ],
            'not_found_message': "I couldn't find detailed hostel information. Please contact the Hostel Administration."
        },
        'about': {
            'required_patterns': [
                r'\babout\b', r'\bhistory\b', r'\bestablish', r'\bfounding\b', 
                r'\bsponsored\b', r'\bvision\b', r'\bmission\b', r'\bphilosophy\b',
                r'\bvallurupalli\b', r'\bfull\s*name\b'
            ],
            'exclude_patterns': [
                r'ED\s+Cell',
                r'(?:The\s+)?[Dd]epartment\s+(?:of\s+\w+\s+)?(?:was|is)\s+(?:established|founded)',
                r'cell\s+has\s+come\s+into\s+existence',
                r'student\s+chapter',
                r'club\s+was\s+(?:established|founded|started)',
            ],
            'boost_patterns': [
                r'Vallurupalli\s+Nageswara\s+Rao',
                r'Full\s+Name\s*:',
                r'Institute\s+History',
                r'Sponsored\s+by',
            ],
            'not_found_message': "I couldn't find detailed information about the institution. Please check the About Us page on the college website."
        },
    }
    
    # Default verification for unknown question types (general queries)
    DEFAULT_VERIFICATION = {
        'required_patterns': [],  # No specific requirements for general queries
        'not_found_message': NO_INFO_RESPONSE
    }
    
    def __init__(self, knowledge_index: Optional[KnowledgeIndex] = None):
        self.index = knowledge_index or get_knowledge_index()
        self.config = RAG_CONFIG
        self._llm = None  # Lazy-loaded LLM generator
        self._use_llm = True  # Toggle for LLM generation
    
    def _get_llm(self):
        """Get or create the LLM generator (lazy loading)."""
        if self._llm is None and self._use_llm:
            try:
                from modules.llm_generator import LLMGenerator
                # Use default initialization (HF router with openai/gpt-oss-20b:groq)
                self._llm = LLMGenerator()
                if not self._llm.is_available():
                    print("[RAG ENGINE] LLM not available, falling back to extraction")
                    self._llm = None
            except Exception as e:
                print(f"[RAG ENGINE] Failed to load LLM: {e}")
                self._llm = None
        return self._llm
    
    def set_index(self, knowledge_index: KnowledgeIndex):
        """Set or update the knowledge index."""
        self.index = knowledge_index
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question being asked."""
        query_lower = query.lower()
        
        # Priority patterns - check these FIRST before general pattern matching
        # This handles cases where queries match multiple types (e.g., "when was established" 
        # could match both 'schedule' (when) and 'about' (establish))
        PRIORITY_OVERRIDES = {
            # MUST check leadership FIRST — HOD/head/chairman queries frequently mismatch
            # because expanded queries contain words that trigger other types
            # (e.g., "Internet of Things" -> 'intern' -> wrong 'placement' type)
            'principal': [
                r'\bhod\b',
                r'\bhead\s+of\s+(?:department|dept|iot|aiml|cse|ece|eee|mech|it)\b',
                r'\bwho\s+is\s+(?:the\s+)?(?:principal|chairman|director|dean)\b',
                r'\bprincipal\s+of\b',
                r'\bchairman\s+of\b',
            ],
            # If query contains establishment keywords, it's an 'about' query even if it has 'when'
            'about': [r'\bestablish', r'\bfound(?:ed)?(?:\s+in)?\b', r'\bstarted\s+in\b', r'\binception\b', r'\borigin\b'],
            # If query asks about designation/department/experience of someone, it's a faculty query
            'faculty': [
                r'\bdesignation\s+of\b', r'\bdepartment\s+of\s+(?!computer|cse|ece|eee|mech|civil|auto)',
                r'\bexperience\s+of\b', r'\bqualification\s+of\b',
                r'\btell\s+me\s+about\s+(?:mr|mrs|ms|dr|prof)',
                r'\bfaculty\b.*\b(?:placement|training)\b',  # "faculty list of placement dept"
                r'\b(?:placement|training)\b.*\bfaculty\b',  # "placement department faculty"
            ],
        }
        
        # Check priority overrides first
        for priority_type, priority_patterns in PRIORITY_OVERRIDES.items():
            for pattern in priority_patterns:
                if re.search(pattern, query_lower):
                    print(f"[QUESTION TYPE] Priority override: '{priority_type}' (matched: {pattern})")
                    return priority_type
        
        # Standard pattern matching
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return q_type
        
        return 'general'
    
    def _create_citations(self, results: List[Tuple[KnowledgeItem, float]]) -> List[Citation]:
        """Create citation objects from search results."""
        citations = []
        seen_sources = set()
        
        for item, score in results:
            # Avoid duplicate citations from same source
            source_key = f"{item.source_type}:{item.source_url}"
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            citation = Citation(
                source_type=item.source_type,
                source_name=item.source_name,
                source_url=item.source_url,
                page_number=item.page_number,
                relevance_score=score,
                snippet=item.content[:200] + "..." if len(item.content) > 200 else item.content
            )
            citations.append(citation)
        
        return citations
    
    def _build_context(self, results: List[Tuple[KnowledgeItem, float]], question_type: str = 'general') -> str:
        """
        Build context string from search results.
        Applies sentence-level filtering to remove irrelevant noise.
        """
        # Get filtering rules
        rules = self.ANSWER_VERIFICATION_RULES.get(question_type, self.DEFAULT_VERIFICATION)
        exclude_patterns = rules.get('exclude_patterns', [])
        
        context_parts = []
        total_length = 0
        max_length = self.config['max_context_length']
        
        for item, score in results:
            if total_length >= max_length:
                break
            
            content = item.content
            
            # SENTENCE-LEVEL FILTERING
            # Split into sentences to remove specific irrelevant lines (like ED Cell details)
            if exclude_patterns:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                filtered_sentences = []
                for sent in sentences:
                    # Skip sentence if it matches any exclude pattern
                    if any(re.search(p, sent, re.IGNORECASE) for p in exclude_patterns):
                        continue
                    filtered_sentences.append(sent)
                content = " ".join(filtered_sentences)
            
            # CLEAN MARKDOWN TABLES for better LLM parsing
            # Convert raw pipe-separated table rows into readable text
            if '|' in content:
                cleaned_lines = []
                table_headers = []
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Skip separator rows like |---|---|
                    if re.match(r'^\|[\s\-:]+\|', line):
                        continue
                    if '|' in line:
                        # Parse table row
                        cells = [c.strip() for c in line.split('|') if c.strip()]
                        # Clean markdown artifacts from cells
                        clean_cells = []
                        for cell in cells:
                            # Remove markdown images [![](url)]
                            cell = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', cell)
                            # Remove markdown links but keep text [text](url)
                            cell = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cell)
                            # Remove bold markers
                            cell = re.sub(r'\*\*(.*?)\*\*', r'\1', cell)
                            # Remove escaped characters
                            cell = cell.replace('\\_', '_')
                            cell = cell.strip()
                            if cell and len(cell) > 1:
                                clean_cells.append(cell)
                        if clean_cells:
                            # Check if this looks like a header row
                            if not table_headers and any(h in ' '.join(clean_cells).lower() for h in ['s.no', 'name', 'designation', 'department']):
                                table_headers = clean_cells
                            else:
                                if table_headers and len(clean_cells) >= 3:
                                    # Map headers to values
                                    pairs = []
                                    for i, val in enumerate(clean_cells):
                                        if i < len(table_headers):
                                            pairs.append(f"{table_headers[i]}: {val}")
                                        else:
                                            pairs.append(val)
                                    cleaned_lines.append(', '.join(pairs))
                                # Guess headers for standard faculty lists (S.No | ID | Name | Desig | ...)
                                elif not table_headers and len(clean_cells) >= 4 and re.match(r'^\d+$', clean_cells[0]):
                                    # Likely [S.No, ID, Name, Designation, ...]
                                    try:
                                        pairs = [
                                            f"S.No: {clean_cells[0]}",
                                            f"ID: {clean_cells[1] if len(clean_cells)>1 else ''}",
                                            f"Name: {clean_cells[2] if len(clean_cells)>2 else ''}",
                                            f"Designation: {clean_cells[3] if len(clean_cells)>3 else ''}",
                                            f"Department: {item.metadata.get('dept', 'OTHER')}", # Use metadata tag instead of hardcoded default
                                        ]
                                        # Add remaining cells as raw text
                                        if len(clean_cells) > 5:
                                            pairs.append(" ".join(clean_cells[5:]))
                                        cleaned_lines.append(', '.join(pairs))
                                    except:
                                        cleaned_lines.append(', '.join(clean_cells))
                                else:
                                    cleaned_lines.append(', '.join(clean_cells))
                    else:
                        cleaned_lines.append(line)
                content = '\n'.join(cleaned_lines)
            

            if not content.strip():
                continue
                
            # Add source attribution
            dept_tag = ""
            if item.metadata and item.metadata.get('dept'):
                dept_tag = f" [Department: {item.metadata.get('dept')}]"
                
            if item.source_type == 'pdf':
                header = f"[From {item.source_name}"
                if item.page_number:
                    header += f", Page {item.page_number}"
                header += f"]{dept_tag}"
            else:
                header = f"[From webpage: {item.source_name}]{dept_tag}"
            
            chunk = f"{header}\n{content}"
            
            if total_length + len(chunk) <= max_length:
                context_parts.append(chunk)
                total_length += len(chunk)
            else:
                # Add partial content
                remaining = max_length - total_length
                if remaining > 100:
                    context_parts.append(chunk[:remaining] + "...")
                break
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, question_type: str) -> Tuple[str, float]:
        """
        Generate answer from context using semantic similarity.
        Uses the shared embedding model to find the most semantically relevant sentences.
        """
        try:
            if not context.strip():
                return self.NO_INFO_RESPONSE, 0.0
            
            # Clean context from webpage artifacts
            context = self._clean_context(context)
            
            # Split context into sentences
            # More robust regex for sentence splitting preserving names with initials like "Dr. B. Chennakesava"
            sentences = re.split(r'(?<!\b[A-Z]\.)(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
            
            # Expanded junk patterns for aggressive filtering
            JUNK_PATTERNS = [
                'click here', 'announcement', 'read more', 'skip to content',
                'menu', 'home', 'contact us', 'copyright', 'all rights reserved',
                'login', 'sign in', 'register', 'subscribe', 'navigation',
                'breadcrumb', 'footer', 'header', 'sidebar', 'share',
                'follow us', 'social media', 'powered by', 'cookie policy',
                'privacy policy', 'terms of service', 'back to top',
                'search', 'download', 'print', 'bookmark'
            ]
            
            # Compile regex pattern with word boundaries for junk matching
            import re
            junk_regex = re.compile(
                r'\b(?:' + '|'.join(map(re.escape, JUNK_PATTERNS)) + r')\b', 
                re.IGNORECASE
            )
            
            candidates = []
            for sent in sentences:
                clean_sent = sent.strip()
                
                # Skip if too short
                if len(clean_sent) < 20:
                    continue
                
                # Skip source headers
                if clean_sent.startswith('['):
                    continue
                
                # Skip page numbers
                if re.match(r'^Page \d+$', clean_sent):
                    continue
                
                # Skip sentences with junk patterns using word boundaries
                if junk_regex.search(clean_sent):
                    continue
                
                # Skip sentences that are mostly navigation (short with common words)
                word_count = len(clean_sent.split())
                if word_count < 5 and any(word in clean_sent.lower() for word in ['menu', 'home', 'here', 'click']):
                    continue
                
                candidates.append(clean_sent)
            
            if not candidates:
                return self.NO_INFO_RESPONSE, 0.0
                
            # Use the existing embedding model if available
            scored_sentences = []
            if self.index and hasattr(self.index, 'model'):
                try:
                    # Encode query and candidates
                    # Wrap query in list to ensure 2D output (1, D)
                    query_emb = self.index.model.encode([query], convert_to_numpy=True)
                    cand_embs = self.index.model.encode(candidates, convert_to_numpy=True)
                    
                    # Compute cosine similarities
                    from sentence_transformers.util import cos_sim
                    # cos_sim returns Tensor (1, M) -> [0] gives Tensor (M) -> tolist() gives list of floats
                    scores = cos_sim(query_emb, cand_embs)[0].tolist()
                    
                    # Boost scores for sentences with numerical/factual data
                    for i, sent in enumerate(candidates):
                        # Boost if contains numbers/amounts (relevant for fee, date queries)
                        if re.search(r'(\d+|Rs\.|₹|INR|\d{4}-\d{2}-\d{2})', sent):
                            scores[i] += 0.1
                        # Boost if contains question-type keywords
                        if question_type in ['fee', 'cost', 'price'] and any(word in sent.lower() for word in ['fee', 'tuition', 'cost', 'amount', 'charges']):
                            scores[i] += 0.05
                            
                        # Boost for 'about' (establishment) questions
                        if question_type == 'about' and any(word in sent.lower() for word in ['established', 'founded', 'started in', 'inception']):
                            scores[i] += 0.15  # Strong boost for founding info
                    
                    scored_sentences = list(zip(candidates, scores))
                except Exception as e:
                    print(f"[ERROR] Model encoding failed: {e}")
                    # Fallback to simple overlap
                    scored_sentences = self._score_by_overlap(query, candidates)
            else:
                 # Fallback to simple overlap
                scored_sentences = self._score_by_overlap(query, candidates)
                
            # Filter by minimum relevance threshold (raised from 0.25 to 0.35 for better precision)
            relevant_sentences = [s for s in scored_sentences if s[1] > 0.35]
            
            if not relevant_sentences:
               return self.NO_INFO_RESPONSE, 0.0
               
            # Sort by relevance
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Pick top sentences (top 3-5)
            top_sentences = []
            seen_content = set()
            
            total_score = 0
            count = 0
            
            for sent, score in relevant_sentences[:5]:
                # Deduplicate near-identical sentences
                sent_key = sent[:40].lower()
                if sent_key in seen_content:
                    continue
                    
                top_sentences.append(sent)
                seen_content.add(sent_key)
                total_score += score
                count += 1
                
            if not top_sentences:
                return self.NO_INFO_RESPONSE, 0.0
    
            # Construct final answer
            answer = " ".join(top_sentences)
            
            # Clean up formatting
            answer = answer.replace('\n', ' ').strip()
            answer = re.sub(r'\s+', ' ', answer)
            
            avg_confidence = total_score / count if count > 0 else 0.0
            # Normalize confidence to 0-1 range
            confidence = min(max(avg_confidence, 0.1), 1.0)
            
            # Validate answer quality before returning
            if not self._is_quality_answer(answer, confidence, question_type):
                print(f"[RAG ENGINE] Answer rejected due to low quality. Confidence: {confidence:.2f}")
                return self.NO_INFO_RESPONSE, 0.0
            
            # Post-process answer for cleaner output
            answer = self._post_process_answer(answer, question_type)
            
            return answer, confidence
            
        except Exception as e:
            print(f"[CRITICAL ERROR] _generate_answer crashed: {e}")
            import traceback
            traceback.print_exc()
            return self.NO_INFO_RESPONSE, 0.0
    
    def _post_process_answer(self, answer: str, question_type: str) -> str:
        """
        Post-process answer for better formatting and readability.
        Cleans up raw extracted text into human-readable response.
        """
        # 1. Fix broken words (common PDF extraction issue)
        # e.g., "Mi ent" → remove, "B. Tech" → "B.Tech"
        answer = re.sub(r'\b([A-Z])\.\s+([A-Z][a-z])', r'\1.\2', answer)  # B. Tech → B.Tech
        answer = re.sub(r'\bMi\s+ent\b', '', answer, flags=re.IGNORECASE)  # Remove "Mi ent" artifact
        
        # 2. Remove standalone fragments (1-2 char words that don't make sense)
        words = answer.split()
        cleaned_words = []
        for i, word in enumerate(words):
            # Skip standalone single letters (except I, a, A)
            if len(word) == 1 and word.lower() not in ['i', 'a', '&']:
                continue
            # Skip if word is just punctuation
            if re.match(r'^[^\w\s]+$', word):
                continue
            cleaned_words.append(word)
        answer = ' '.join(cleaned_words)
        
        # 3. Remove incomplete sentences at the start
        # If answer starts with lowercase or partial word, find first proper sentence
        if answer and answer[0].islower():
            # Find first capital letter that starts a sentence
            match = re.search(r'[.!?]\s+([A-Z])', answer)
            if match:
                answer = answer[match.end()-1:]
        
        # 4. Remove trailing incomplete sentences
        # If doesn't end with proper punctuation, cut at last complete sentence
        if answer and answer[-1] not in '.!?':
            last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
            if last_period > len(answer) * 0.5:  # Only cut if we keep at least half
                answer = answer[:last_period + 1]
        
        # 5. Standardize formatting
        answer = re.sub(r'\s+', ' ', answer)  # Multiple spaces to single
        answer = re.sub(r'\s+([.,!?])', r'\1', answer)  # Remove space before punctuation
        answer = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', answer)  # Add space after punctuation
        
        # 6. Question-type specific formatting
        if question_type == 'fee':
            # Format currency nicely: Rs.130000 → Rs. 1,30,000
            answer = re.sub(r'Rs\.?\s*(\d+)', lambda m: f"Rs. {int(m.group(1)):,}", answer)
            # Ensure rupee symbol has space: ₹130000 → ₹ 1,30,000
            answer = re.sub(r'₹\s*(\d+)', lambda m: f"₹ {int(m.group(1)):,}", answer)
        
        if question_type == 'contact':
            # Format phone numbers: add dashes
            answer = re.sub(r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3', answer)
        
        # 7. Ensure proper capitalization at start
        if answer:
            answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer.upper()
        
        # 8. Final cleanup
        answer = answer.strip()
        
        print(f"[POST-PROCESS] Answer cleaned: {len(answer)} chars")
        return answer
    
    def _clean_context(self, context: str) -> str:
        """Remove navigation artifacts and junk from context."""
        # Remove common webpage patterns
        context = re.sub(r'Click\s+here\s*', '', context, flags=re.IGNORECASE)
        context = re.sub(r'\[From\s+webpage:.*?\]', '', context)
        context = re.sub(r'Announcement\s*\n', '', context)
        context = re.sub(r'Read\s+more\s*', '', context, flags=re.IGNORECASE)
        
        # Remove isolated navigation words
        context = re.sub(r'\b(Home|Menu|Login|Sign\s+in|Register)\b(?![a-z])', '', context, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        context = re.sub(r'\n\s*\n', '\n', context)
        context = re.sub(r'\s+', ' ', context)
        
        return context.strip()
    
    def _is_quality_answer(self, answer: str, confidence: float, question_type: str) -> bool:
        """Check if answer is human-readable and useful."""
        # Reject if too short
        words = answer.split()
        if len(words) < 5:
            return False
        
        # Count junk/navigation words
        junk_words = ['click', 'here', 'announcement', 'read', 'more', 'menu', 'home', 
                      'login', 'sign', 'register', 'mi', 'ent']
        junk_count = sum(1 for word in words if word.lower() in junk_words)
        junk_ratio = junk_count / len(words)
        
        # Reject if mostly junk
        if junk_ratio > 0.25:  # 25% junk = bad answer
            return False
        
        # Check for factual content (numbers, specific terms)
        has_numbers = bool(re.search(r'\d+', answer))
        has_specific_terms = any(term in answer.lower() for term in 
                                ['fee', 'rs', '₹', 'tuition', 'semester', 'year', 'department', 'program'])
        
        # For low confidence answers, require factual content
        if confidence < 0.5 and not (has_numbers or has_specific_terms):
            return False
        
        # Check if answer is just fragments (many single-char words or odd fragments)
        fragment_count = sum(1 for word in words if len(word) <= 2)
        if fragment_count / len(words) > 0.3:  # 30% fragments
            return False
        
        return True
    
    def _verify_chunks(self, results: List[Tuple[KnowledgeItem, float]], question_type: str, query: str = "", is_cse_query: bool = False) -> Tuple[List[Tuple[KnowledgeItem, float]], str]:
        """
        Verify retrieved chunks match the expected answer type AND program type.
        
        Features:
        1. CSE Dept Boosting: If CSE query, boost CSE chunks and penalize others.
        2. FAQ Boosting: Prioritize FAQ/custom markdown sources over PDF sources.
        3. Program Filtering: If query is for B.Tech, reject M.Tech chunks (and vice versa).
        4. Content Verification: Ensure chunks match the question type (e.g. fee queries need money).
        """
        # ========================================================
        # FAQ SOURCE BOOSTING (Prioritize FAQ/custom sources)
        # ========================================================
        # Boost chunks from FAQ.md, faq.md, 000_contact_info.md, etc.
        faq_boosted_results = []
        for item, score in results:
            source_lower = item.source_name.lower()
            url_lower = (item.source_url or "").lower()
            
            # Check if this is a FAQ/custom source
            is_faq_source = any([
                'faq' in source_lower,
                'contact' in source_lower,
                'custom/' in url_lower,
                source_lower.startswith('vnrvjiet contact'),
            ])
            
            if is_faq_source:
                # Boost FAQ sources by 0.15 (significant boost)
                boosted_score = min(score + 0.15, 1.0)
                print(f"[VERIFY BOOST] FAQ source boosted: {item.source_name[:40]}... +0.15")
                faq_boosted_results.append((item, boosted_score))
            else:
                faq_boosted_results.append((item, score))
        
        # Re-sort by boosted scores (highest first)
        results = sorted(faq_boosted_results, key=lambda x: x[1], reverse=True)
        
        # Compute query_lower once for all filtering stages
        query_lower = query.lower()
        
        # ========================================================
        # PROGRAM FILTERING (B.Tech vs M.Tech)
        # ========================================================
        target_program = None
        
        # Detect target program from query
        if re.search(r'\bb\.?tech\b|\bunder\s*grad', query_lower):
            target_program = 'btech'
        elif re.search(r'\bm\.?tech\b|\bpost\s*grad', query_lower):
            target_program = 'mtech'
            
        filtered_results = []
        if target_program:
            print(f"[VERIFY] Target Program Detected: {target_program.upper()}")
            
            for item, score in results:
                # Check for CONFLICTING program mentions in source name or content
                # We simply check if the chunk explicitly mentions the OTHER program
                # and DOES NOT mention the target program
                
                content_lower = (item.source_name + " " + item.content).lower()
                
                is_btech_content = bool(re.search(r'\bb\.?tech\b|\bb\.\s*tech', content_lower))
                is_mtech_content = bool(re.search(r'\bm\.?tech\b|\bm\.\s*tech', content_lower))
                
                if target_program == 'btech':
                    # Reject if explicitly M.Tech and NOT B.Tech
                    if is_mtech_content and not is_btech_content:
                        print(f"[VERIFY FILTER] Rejected M.Tech chunk for B.Tech query: {item.source_name}")
                        continue
                elif target_program == 'mtech':
                    # Reject if explicitly B.Tech and NOT M.Tech
                    if is_btech_content and not is_mtech_content:
                        print(f"[VERIFY FILTER] Rejected B.Tech chunk for M.Tech query: {item.source_name}")
                        continue
                
                filtered_results.append((item, score))
            
            results = filtered_results # Continue with filtered list
        
        # ========================================================
        # FEE TYPE FILTERING (Domestic vs International)
        # ========================================================
        # For fee queries, filter out International/NRI/Foreign fees unless explicitly requested
        if question_type == 'fee':
            # Check if user is asking for international/foreign/NRI fees
            asking_for_international = bool(re.search(
                r'\binternational\b|\bforeign\b|\bnri\b|\boverseas\b|\babroad\b',
                query_lower
            ))
            
            if not asking_for_international:
                print(f"[VERIFY] Fee query without 'international' keyword - filtering out International/NRI/USD chunks")
                
                fee_filtered = []
                for item, score in results:
                    content_combined = (item.source_name + " " + item.content).lower()
                    
                    # Check if chunk is about International/NRI/Foreign fees
                    is_international_fee = bool(re.search(
                        r'international\s+student|nri\s+fee|foreign\s+national|usd\s*\$|\$\s*\d+|overseas',
                        content_combined
                    ))
                    
                    if is_international_fee:
                        print(f"[VERIFY FILTER] Removed International fee chunk: {item.source_name[:50]}...")
                        continue
                        
                    fee_filtered.append((item, score))
                
                results = fee_filtered
                
                if not results:
                    print(f"[VERIFY WARNING] All chunks were International fees - query may need broader context")
            else:
                print(f"[VERIFY] User explicitly asked for International fees - keeping all results")
            
        # ========================================================
        # CONTENT VERIFICATION
        # ========================================================
        # Get verification rules for this question type
        rules = self.ANSWER_VERIFICATION_RULES.get(question_type, self.DEFAULT_VERIFICATION)
        required_patterns = rules['required_patterns']
        exclude_patterns = rules.get('exclude_patterns', [])
        boost_patterns = rules.get('boost_patterns', [])  # Initialize boost_patterns
        not_found_message = rules['not_found_message']
        
        # If no specific patterns required (general query), keep all chunks
        if not required_patterns:
            print(f"[VERIFY] Question type '{question_type}' has no specific requirements, keeping all chunks")
            return results, self.NO_INFO_RESPONSE
        
        # Filter chunks that match at least one required pattern
        verified_chunks = []
        
        for item, score in results:
            content = item.content
            content_lower = content.lower()
            
            # First check exclusion patterns - reject if matches any
            excluded = False
            for pattern in exclude_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    print(f"[VERIFY EXCLUDED] Chunk from '{item.source_name}' matches exclusion: {pattern[:40]}...")
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Check if chunk matches any required pattern
            matches_pattern = False
            
            # Source-name whitelist: if the source name clearly matches the query type,
            # auto-pass it (e.g., a chunk from "2021_25_batch_CSE_Placements" for a placement query)
            source_name_lower = item.source_name.lower()
            SOURCE_WHITELISTS = {
                'placement': [r'placement', r'placed', r'recruit'],
                'fee': [r'fee', r'tuition'],
                'admission': [r'admission', r'eligib'],
            }
            whitelist_patterns = SOURCE_WHITELISTS.get(question_type, [])
            for wp in whitelist_patterns:
                if re.search(wp, source_name_lower):
                    matches_pattern = True
                    print(f"[VERIFY OK] Chunk from '{item.source_name}' auto-passed (source name matches '{question_type}')")
                    break
            
            if not matches_pattern:
                for pattern in required_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        matches_pattern = True
                        print(f"[VERIFY OK] Chunk from '{item.source_name}' matches pattern: {pattern[:40]}...")
                        break
            
            if matches_pattern:
                # Apply boost to score if matches boost patterns
                boosted_score = score
                for pattern in boost_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        boost = 0.15  # Significant boost
                        boosted_score = min(score + boost, 1.0)
                        print(f"[VERIFY BOOST] Chunk from '{item.source_name}' boosted +{boost} for: {pattern[:40]}...")
                        break
                
                verified_chunks.append((item, boosted_score))
            else:
                print(f"[VERIFY X] Chunk from '{item.source_name}' REJECTED - no {question_type} content found")
        
        # Re-sort by boosted scores
        if boost_patterns:
            verified_chunks.sort(key=lambda x: x[1], reverse=True)
            print(f"[VERIFY] Chunks re-sorted by boosted scores")
        
        print(f"[VERIFY] {len(verified_chunks)}/{len(results)} chunks passed verification for '{question_type}' query")
        
        return verified_chunks, not_found_message

    def _score_by_overlap(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Fallback scoring using simple word overlap."""
        query_words = set(query.lower().split())
        scored = []
        for sent in candidates:
            sent_words = set(sent.lower().split())
            if not sent_words:
                continue
            overlap = len(query_words & sent_words)
            score = overlap / len(query_words) if query_words else 0
            scored.append((sent, score))
        return scored
    
    def _extract_leadership_answer(self, query: str, results: List[Tuple[KnowledgeItem, float]]) -> Optional[str]:
        """
        Direct extraction for leadership queries (chairman, principal, director, dean, hod).
        Extracts names from structured table data without relying on LLM.
        Returns a clean answer string, or None if extraction fails.
        
        Strategies (tried in order):
        1. Table row: role keyword in category column (3rd column)
        2. Table row: role keyword embedded in name column (2nd column)
        3. Plain text: "Role: Dr. Name" or "Dr. Name, Role"
        4. Profile page: dedicated page (e.g. /principal/) → first named person
        """
        query_lower = query.lower()
        
        # Determine which role we're looking for
        role_keywords = {
            'chairman': ['chairman', 'chairperson'],
            'principal': ['principal'],
            'director': ['director'],
            'dean': ['dean'],
            'hod': ['hod', 'head of department', 'head'],
        }
        
        target_role = None
        for role, keywords in role_keywords.items():
            if any(kw in query_lower for kw in keywords):
                target_role = role
                break
        
        if not target_role:
            return None
            
        print(f"[EXTRACT] Looking for '{target_role}' in {len(results)} retrieved chunks...")
        
        # --------------------------------------------------------
        # Strategy 0: BOOSTED NAME MATCH (highest priority)
        # Check if any boost_patterns from ANSWER_VERIFICATION_RULES
        # match in the retrieved chunks. This handles cases where the
        # governance page lists the person with a different title
        # (e.g., "Director" instead of "Principal").
        # NOTE: HOD/dean/director all use 'principal' rules since all
        # leadership boost patterns are stored there (no separate 'hod' key).
        # --------------------------------------------------------
        rules_key = target_role if target_role in self.ANSWER_VERIFICATION_RULES else 'principal'
        question_rules = self.ANSWER_VERIFICATION_RULES.get(rules_key, self.DEFAULT_VERIFICATION)
        boost_patterns = question_rules.get('boost_patterns', [])
        
        for item, score in results:
            content = item.content
            for bp in boost_patterns:
                if re.search(bp, content, re.IGNORECASE):
                    # Found the boosted pattern! Extract full name with honorific.
                    # Allow optional middle initials between honorific and name (e.g. Dr. Y. Padma Sai)
                    full_name_re = re.compile(
                        r'((?:Dr\.?|Prof\.?|Sri|Smt\.?)\s+(?:[A-Z]\.?\s+)*' + bp + r')',
                        re.IGNORECASE
                    )
                    fm = full_name_re.search(content)
                    if fm:
                        name = fm.group(1).strip().rstrip(',.')
                        # Verify it looks like a person name (has uppercase + lowercase letters)
                        if re.search(r'[A-Z][a-z]', name):
                            print(f"[EXTRACT] Strategy 0 — boosted name match: {name}")
                            return self._format_leadership_answer(target_role, name)
        
        for item, score in results:
            content = item.content
            
            # --------------------------------------------------------
            # Strategy 1: Role keyword in CATEGORY column (3rd column)
            # Pattern: | 1 | **Sri D. Suresh Babu** | **Chairman, Nominated by Society, VJ** |
            # --------------------------------------------------------
            table_pattern = re.compile(
                r'\|\s*\d+\s*\|\s*\*{0,2}([^|*]+?)\*{0,2}\s*\|\s*\*{0,2}([^|*]*?' + target_role + r'[^|*]*?)\*{0,2}\s*\|',
                re.IGNORECASE
            )
            matches = table_pattern.findall(content)
            
            if matches:
                name = matches[0][0].strip()
                role_desc = matches[0][1].strip()
                # Clean trailing commas/spaces from name
                name = re.sub(r'[,\s]+$', '', name)
                print(f"[EXTRACT] Strategy 1 — table category column: {name} — {role_desc}")
                return self._format_leadership_answer(target_role, name, role_desc)
            
            # --------------------------------------------------------
            # Strategy 2: Role keyword EMBEDDED in NAME column (2nd column)
            # Pattern: | 18 | **Dr. C. D. Naidu, Principal** | **Member Secretary** |
            # --------------------------------------------------------
            embedded_pattern = re.compile(
                r'\|\s*\d+\s*\|\s*\*{0,2}((?:Dr\.?|Prof\.?|Sri|Smt\.?|Mr\.?|Mrs\.?)\s+[^|*]+?),\s*' + target_role + r'[^|]*?\*{0,2}\s*\|',
                re.IGNORECASE
            )
            embedded_match = embedded_pattern.search(content)
            
            if embedded_match:
                name = embedded_match.group(1).strip()
                name = re.sub(r'[,\s]+$', '', name)
                print(f"[EXTRACT] Strategy 2 — embedded in name column: {name}")
                return self._format_leadership_answer(target_role, name)
            
            # --------------------------------------------------------
            # Strategy 3: Plain text patterns
            # Pattern: "Chairman: Dr. Name" or "Dr. Name, Chairman"
            # --------------------------------------------------------
            text_patterns = [
                re.compile(r'(?:' + target_role + r')\s*[,:—\-]\s*(?:Dr\.?|Prof\.?|Sri|Smt\.?|Mr\.?|Mrs\.?)\s+([A-Z][A-Za-z\s\.]+)', re.IGNORECASE),
                re.compile(r'(?:Dr\.?|Prof\.?|Sri|Smt\.?|Mr\.?|Mrs\.?)\s+([A-Z][A-Za-z\s\.]+?)\s*[,—\-]\s*(?:' + target_role + r')', re.IGNORECASE),
            ]
            
            for pattern in text_patterns:
                match = pattern.search(content)
                if match:
                    name = match.group(1).strip().rstrip(',.')
                    print(f"[EXTRACT] Strategy 3 — plain text: {name}")
                    return self._format_leadership_answer(target_role, name)
            
            # --------------------------------------------------------
            # Strategy 4: Profile page extraction
            # If the source is a dedicated page (e.g. /principal/, /dean-iqac/)
            # Extract the first Dr./Prof./Sri name from the content body
            # --------------------------------------------------------
            source_lower = (item.source_name or '').lower()
            url_lower = (item.source_url or '').lower()
            
            if target_role in source_lower or target_role in url_lower:
                # This chunk is from a dedicated role page — extract first named person
                # Capture the FULL name including honorific (Dr., Prof., Sri, etc.)
                name_pattern = re.compile(
                    r'((?:Dr\.?|Prof\.?|Sri|Smt\.?)\s+[A-Z][A-Za-z\.\s]{3,40}?)(?:\s+(?:graduated|is|has|was|served|worked|joined|received))',
                    re.IGNORECASE
                )
                name_match = name_pattern.search(content)
                if name_match:
                    name = name_match.group(1).strip().rstrip(',.')
                    print(f"[EXTRACT] Strategy 4 — profile page ({item.source_name}): {name}")
                    return self._format_leadership_answer(target_role, name)
                
                # Fallback: find "working as **Principal** in VNR" pattern and look backward
                role_mention = re.search(
                    r'((?:Dr\.?|Prof\.?|Sri)\s+[A-Z][A-Za-z\s\.]+?)(?:\s+(?:is|has been|was)\s+(?:the\s+)?(?:currently\s+)?(?:working\s+as\s+)?\*{0,2}' + target_role + r')',
                    content, re.IGNORECASE
                )
                if role_mention:
                    name = role_mention.group(1).strip().rstrip(',.')
                    print(f"[EXTRACT] Strategy 4 — role mention in profile page: {name}")
                    return self._format_leadership_answer(target_role, name)
        
        print(f"[EXTRACT] Could not extract {target_role} name from chunks")
        return None
    
    def _format_leadership_answer(self, role: str, name: str, role_desc: str = None) -> str:
        """Format a clean leadership answer."""
        role_title = role.title()
        if role == 'chairman' and role_desc:
            return f"The Chairman of VNRVJIET is **{name}**. {name} is the {role_desc} of the Governing Council."
        elif role == 'dean' and role_desc:
            return f"The Dean of VNRVJIET is **{name}** ({role_desc})."
        else:
            return f"The {role_title} of VNRVJIET is **{name}**."

    def _get_hardcoded_override(self, query: str) -> Optional[str]:
        """
        Provides direct answers for known leadership/HOD queries where dynamic
        retrieval is unreliable (names buried in faculty tables far below top-20).
        
        UPDATE THIS TABLE when staff changes occur.
        Last updated: March 2026
        """
        query_lower = query.lower()
        
        # -----------------------------------------------------------------------
        # HOD LOOKUP TABLE  (department keyword -> (name, dept_full_name))
        # -----------------------------------------------------------------------
        HOD_TABLE = {
            # ECE
            frozenset(['ece', 'electronics and communication', 'electronics & communication']): 
                ('Dr. Y. Padma Sree', 'Electronics and Communication Engineering'),
            # EEE
            frozenset(['eee', 'electrical and electronics', 'electrical & electronics']):
                ('Dr. Poonam Upadhyay', 'Electrical and Electronics Engineering'),
            # CSE (plain)
            frozenset(['cse', 'computer science']):
                ('Dr. Vadlana Baby', 'Computer Science and Engineering'),
            # CSBS
            frozenset(['csbs', 'computer science and business', 'computer science & business', 'business systems']):
                ('Dr. Vadlana Baby', 'Computer Science and Business Systems'),
            # CSE-IoT
            frozenset(['iot', 'internet of things', 'cse-iot', 'cse iot']):
                ('Dr. Sagar Yeruva', 'CSE (Internet of Things)'),
            # CSE-AIML
            frozenset(['aiml', 'ai & ml', 'ai and ml', 'artificial intelligence', 'machine learning', 'cse-aiml']):
                ('Dr. Sagar Yeruva', 'CSE (Artificial Intelligence & Machine Learning)'),
            # Mechanical
            frozenset(['mechanical', 'mech']):
                ('Prof. K. Narayana Rao', 'Mechanical Engineering'),
            # Civil
            frozenset(['civil']):
                ('Prof. P. Srinivas Rao', 'Civil Engineering'),
            # IT
            frozenset(['information technology', ' it department', 'dept of it']):
                ('Dr. M. Sunil Kumar', 'Information Technology'),
        }
        
        # Check if this is a HOD query
        is_hod_query = re.search(r'\b(?:hod|head\s+of\s+(?:department|dept))\b', query_lower)
        if not is_hod_query:
            return None
        
        # Find matching department
        for dept_keywords, (name, dept_name) in HOD_TABLE.items():
            if any(kw in query_lower for kw in dept_keywords):
                return f"The Head of the Department (HOD) of **{dept_name}** at VNRVJIET is **{name}**."

        return None

    def query(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Process a query and generate a grounded response.
        
        Args:
            query: The user's question
            top_k: Number of context items to retrieve
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        print(f"\n[RAG ENGINE] Processing query: '{query[:80]}...'")
        
        if not self.index:
            print("[RAG ENGINE ERROR] Index is None!")
            return RAGResponse(
                query=query,
                answer="Knowledge base not initialized. Please run indexing first.",
                citations=[],
                confidence=0.0,
                grounded=False,
                raw_context=""
            )
        
        top_k = top_k or self.config['top_k']
        min_similarity = self.config['min_similarity']
        
        print(f"[RAG ENGINE] Config: top_k={top_k}, min_similarity={min_similarity}")
        
        # Query Expansion for Full Name queries (fixes retrieval recall issues)
        if re.search(r'full\s*name|stand\s*for|full\s*form|what\s+is\s+vnrvjiet', query, re.IGNORECASE):
             if "Vallurupalli" not in query:
                query += " Vallurupalli Nageswara Rao Vignana Jyothi"
                print(f"[RAG ENGINE] Expanded query with full name keywords")

        # Query Expansion for Programme Count queries
        if re.search(r'how\s+many\s+(?:programmes?|courses?|branches)', query, re.IGNORECASE):
            # Inject exact keywords from the admissions summary to improve retrieval
            query += " offers 17 B.Tech programmes 15 M.Tech programmes 5 Ph.D programmes"
            print(f"[RAG ENGINE] Expanded query with programme count keywords")

        # Leadership Query Expansion (chairman, director, principal, dean, hod)
        # Inject governance-specific keywords to bias FAISS retrieval toward correct pages
        leadership_keywords = {
            r'\bchairman\b': "governing council chairman nominated by society Sri D. Suresh Babu",
            r'\bdirector\b': "director administration advancement Dr. B. Chennakesava Rao",
            r'\bprincipal\b': "principal member secretary Dr. B. Chennakesava Rao institution head VNRVJIET",
            r'\bdean\b': "dean academics administration",
            r'\bhod\s+of\s+eee\b': "Dr. Poonam Upadhyay Professor and Head EEE Electrical Electronics",
            r'\bhod\s+of\s+ece\b': "Professor and Head ECE Electronics Communication Dr. Y. Padma Sai",
            r'\bhod\s+of\s+cse(?![-a-zA-Z])\b': "Dr. Vadlana Baby Associate Professor HOD CSE Computer Science",
            r'\bhod\s+of\s+mech\b': "Professor and Head Mechanical Engineering",
            r'\bhod\s+of\s+it\b': "Professor and Head Information Technology",
            # IoT and AIML HOD — both are Dr. Sagar Yeruva
            r'\bhod\s+of\s+(?:iot|internet\s+of\s+things|cse[-\s]iot)\b': "Dr. Sagar Yeruva Professor and Head CSE-IOT Internet of Things",
            r'\bhod\s+of\s+(?:aiml|ai\s*&?\s*ml|cse[-\s]aiml|artificial\s+intelligence)\b': "Dr. Sagar Yeruva Professor and Head CSE-AIML Artificial Intelligence Machine Learning",
            r'\bhead\s+of\s+(?:iot|internet\s+of\s+things)\b': "Dr. Sagar Yeruva Professor and Head CSE-IOT Internet of Things",
            r'\bhead\s+of\s+(?:aiml|ai\s*&?\s*ml|artificial\s+intelligence)\b': "Dr. Sagar Yeruva Professor and Head CSE-AIML Artificial Intelligence Machine Learning",
            # CSBS HOD — Dr. Vadlana Baby
            r'\bhod\s+of\s+(?:csbs|cse[-\s]bs|computer\s+science.*business|business\s+systems)\b': "Dr. Vadlana Baby Professor and Head CSBS Computer Science Business Systems",
            r'\bhead\s+of\s+(?:csbs|business\s+systems)\b': "Dr. Vadlana Baby Professor and Head CSBS Computer Science Business Systems",
        }
        for lk_pattern, lk_expansion in leadership_keywords.items():
            if re.search(lk_pattern, query, re.IGNORECASE):
                query += f" {lk_expansion}"
                print(f"[RAG ENGINE] Leadership query expansion: added '{lk_expansion}'")
                break  # Only one expansion


        # Acronym Expansion (helps with IIIC, EDC, etc.)
        acronyms = {
            r'\bIIIC\b': "Industry Institute Interaction Cell",
            r'\bED\s*Cell\b': "Entrepreneurship Development Cell",
            r'\bEDC\b': "Entrepreneurship Development Cell",
            r'\bIQAC\b': "Internal Quality Assurance Cell",
            r'\bNSS\b': "National Service Scheme",
            r'\bNCC\b': "National Cadet Corps",
            r'\bExam(?:ination)?\s+Branch\b': "Examination Cell Controller of Examinations",
        }
        for pattern, expansion in acronyms.items():
            if re.search(pattern, query, re.IGNORECASE) and expansion.lower() not in query.lower():
                query += f" ({expansion})"
                print(f"[RAG ENGINE] Expanded acronym: {pattern} -> {expansion}")
        
        # Detect question type
        print(f"[RAG ENGINE] Detecting question type for: '{query}'")
        question_type = self._detect_question_type(query)
        print(f"[RAG ENGINE] Question type detected: {question_type}")

        # QUERY CLASSIFICATION DECISION (fixes retrieval bias)
        query_lower = query.lower()
        is_general_query = any(kw in query_lower for kw in GENERAL_KEYWORDS)
        # Use word-boundary matching for OTHER_BRANCH_KEYWORDS to prevent
        # 'it' from matching 'institution', 'civil' from matching 'civilization', etc.
        is_other_branch_query = any(
            re.search(r'\b' + re.escape(kw) + r'\b', query_lower)
            for kw in OTHER_BRANCH_KEYWORDS
        )
        is_cse_query_raw = any(kw in query_lower for kw in CSE_QUERY_KEYWORDS)

        # Build ChromaDB metadata filters
        where_conditions = []
        is_cse_query = False
        
        # 1. Check for specific CSE Sub-Departments first!
        sub_dept_match = None
        for dept_tag, keywords in CSE_SUB_DEPT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                sub_dept_match = dept_tag
                break
                
        if sub_dept_match and not is_other_branch_query:
            where_conditions.append({"dept": sub_dept_match})
            print(f"[RAG ENGINE] [{sub_dept_match}] SUB-DEPT DETECTED -- using ChromaDB filter 'dept': '{sub_dept_match}'")
        elif is_cse_query_raw and not is_other_branch_query:
            is_cse_query = True
            # Expand to include all CSE sub-departments for general CSE queries
            dept_options = ["CSE"] + list(CSE_SUB_DEPT_KEYWORDS.keys())
            where_conditions.append({"dept": {"$in": dept_options}})
            print(f"[RAG ENGINE] [CSE] GENERAL CSE PRIORITY DETECTED -- using ChromaDB filter 'dept' in {dept_options}")
        elif is_other_branch_query:
            where_conditions.append({"dept": "OTHER"})

        # CRITICAL: For leadership queries (principal, chairman, dean, hod),
        # CLEAR any department filter to ensure profile pages are always retrievable.
        # Check BOTH question_type AND raw query keywords for double safety.
        # Without this, the principal's page could be excluded if the query expansion
        # accidentally triggers a department keyword match.
        is_leadership_query = (
            question_type in ['principal', 'leadership'] or
            re.search(r'\b(?:hod|head\s+of\s+(?:department|dept|iot|aiml|cse|ece|eee|mech|it)|principal|chairman|dean)\b', query_lower)
        )
        if is_leadership_query:
            where_conditions = []  # Remove ALL dept filters for leadership queries
            print(f"[RAG ENGINE] Leadership/HOD query detected -- CLEARING dept filters for unbiased search")

        # Build the final where_filter with $and if multiple conditions
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}
        else:
            where_filter = None

        # DYNAMIC TOP_K STRATEGY (Critical for fixing numeric hallucinations)
        dynamic_k = top_k  # Start with default
        
        if question_type in ['fee', 'numeric']:
            dynamic_k = 5
            print(f"[RAG ENGINE] [FEE] NUMERIC/FEE QUERY DETECTED: Setting top_k=5 to capture fee data")
        elif question_type == 'faculty':
            dynamic_k = 10
            print(f"[RAG ENGINE] [FACULTY] Faculty query: Setting top_k=10 for faculty lookup")
        elif question_type == 'principal':
            dynamic_k = 20
            print(f"[RAG ENGINE] [PRINCIPAL] Leadership query: Setting top_k=20 to capture faculty tables")
        elif question_type in ['placement', 'about', 'facility']:
            dynamic_k = 8
            print(f"[RAG ENGINE] [INFO] Complex query: Setting top_k=8")
        else:
            dynamic_k = 5
            print(f"[RAG ENGINE] [STD] Standard query: Setting top_k=5")
            
        # Override if user manually passed top_k, otherwise use dynamic
        final_k = top_k if top_k != self.config['top_k'] else dynamic_k
        
        final_min_similarity = min_similarity
        if question_type in ['principal', 'placement']:
            final_min_similarity = 0.25 if question_type == 'principal' else 0.35
            print(f"[RAG ENGINE] {question_type.title()} query: Lowering min_similarity to {final_min_similarity}")
        
        # Search for relevant content
        print(f"[RAG ENGINE] Searching index for relevant content (k={final_k}, where={where_filter})...")
        try:
            results = self.index.search(query, top_k=final_k, min_similarity=final_min_similarity, where=where_filter)
            print(f"[RAG ENGINE] Found {len(results)} relevant chunks")
            
            # Log what we actually retrieved
            if results:
                print(f"\n{'='*60}")
                print(f"RETRIEVED CHUNKS FOR: '{query[:60]}...'")
                print(f"{'='*60}")
                for i, (item, score) in enumerate(results[:5], 1):
                    print(f"\n[CHUNK {i}] Similarity: {score:.3f}")
                    print(f"  Source: {item.source_name} ({item.source_type})")
                    if item.page_number:
                        print(f"  Page: {item.page_number}")
                    snippet = item.content[:150].encode('ascii', 'ignore').decode('ascii')
                    print(f"  Content snippet: {snippet}...")
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"[RAG ENGINE ERROR] Search failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if not results:
            print("[RAG ENGINE] No relevant content found")
            # Get type-specific not found message
            rules = self.ANSWER_VERIFICATION_RULES.get(question_type, self.DEFAULT_VERIFICATION)
            not_found_msg = rules.get('not_found_message', self.NO_INFO_RESPONSE)
            return RAGResponse(
                query=query,
                answer=not_found_msg,
                citations=[],
                confidence=0.0,
                grounded=False,
                raw_context=""
            )
        
        # ========================================
        # VERIFICATION LAYER - Answer-Type Aware
        # ========================================
        print(f"\n[VERIFICATION] Applying '{question_type}' verification rules...")
        verified_results, not_found_message = self._verify_chunks(results, question_type, query, is_cse_query=is_cse_query)
        
        # If no chunks passed verification, return type-specific not found message
        if not verified_results:
            print(f"[VERIFICATION] NO chunks passed verification for '{question_type}' query!")
            print(f"[VERIFICATION] Returning: {not_found_message[:60]}...")
            return RAGResponse(
                query=query,
                answer=not_found_message,
                citations=[],
                confidence=0.0,
                grounded=False,
                raw_context=""
            )
        
        print(f"[VERIFICATION] OK: {len(verified_results)} chunks passed verification")
        
        # Use verified results for answer generation
        results = verified_results
        # ========================================
        
        # Build context from VERIFIED results
        print(f"[RAG ENGINE] Building context from verified results...")
        context = self._build_context(results, question_type)
        print(f"[RAG ENGINE] Context built: {len(context)} chars")
        
        # Create citations from verified results
        citations = self._create_citations(results)
        print(f"[RAG ENGINE] Created {len(citations)} citations")
        
        # ========================================
        # LLM ANSWER GENERATION
        # ========================================
        print(f"[RAG ENGINE] Generating answer via LLM...")
        
        # FIRST: Check for hardcoded overrides (e.g. HODs)
        override_answer = self._get_hardcoded_override(query)
        
        if override_answer:
            print(f"[RAG ENGINE] Using hardcoded override for answer: {override_answer}")
            answer = override_answer
            confidence = 1.0
        else:
            # ========================================
            # LEADERSHIP EXTRACTION (BEFORE LLM)
            # For principal/chairman/dean/hod queries, try direct name extraction
            # from verified chunks FIRST. This is more reliable than LLM for
            # structured data like profile pages and governance tables.
            # ========================================
            if question_type in ['principal', 'leadership']:
                print(f"[RAG ENGINE] Trying direct leadership extraction for '{question_type}' query...")
                extracted_answer = self._extract_leadership_answer(query, results)
                if extracted_answer:
                    print(f"[RAG ENGINE] Leadership extraction SUCCESS: {extracted_answer[:80]}...")
                    answer = extracted_answer
                    confidence = 0.95
                else:
                    print(f"[RAG ENGINE] Leadership extraction returned None, falling back to LLM...")
                    extracted_answer = None  # Will fall through to LLM below
            else:
                extracted_answer = None
            
            # If leadership extraction already provided an answer, skip LLM
            if question_type in ['principal', 'leadership'] and extracted_answer:
                pass  # Answer already set above
            else:
                # NORMAL LLM EXECUTION
                try:
                    llm = self._get_llm()
                    
                    if llm and llm.is_available():
                        # Use LLM for natural answer synthesis
                        print(f"[RAG ENGINE] Using {self._llm.model_name} for answer generation...")
                        
                        # Get exclude patterns for sentence-level filtering
                        rules = self.ANSWER_VERIFICATION_RULES.get(question_type, self.DEFAULT_VERIFICATION)
                        exclude_patterns = rules.get('exclude_patterns', [])
                        
                        # Prepare chunks for LLM WITH SENTENCE-LEVEL FILTERING
                        chunks_for_llm = []
                        for item, score in results:
                            content = item.content
                            
                            # Apply sentence-level filtering to remove irrelevant noise
                            if exclude_patterns:
                                sentences = re.split(r'(?<=[.!?])\s+', content)
                                filtered = [s for s in sentences if not any(re.search(p, s, re.IGNORECASE) for p in exclude_patterns)]
                                content = " ".join(filtered)
                            
                            if content.strip():
                                chunks_for_llm.append({
                                    'source_name': item.source_name,
                                    'source_url': item.source_url,
                                    'content': content,
                                    'section': item.metadata.get('section', item.source_name)
                                })
                        
                        # Generate answer with LLM
                        llm_result = llm.generate_answer(query, chunks_for_llm)
                        
                        if llm_result.answer:
                            answer = llm_result.answer
                            # High confidence for LLM-generated answers
                            confidence = 0.85
                            print(f"[RAG ENGINE] LLM answer generated in {llm_result.generation_time:.2f}s")
                        else:
                            # LLM failed - return clean message instead of dumping raw chunks
                            print(f"[RAG ENGINE] LLM returned empty, returning service message...")
                            answer = "I found relevant information but the AI service is temporarily busy. Please try again in a moment."
                            confidence = 0.5
                    else:
                        # Fallback to extraction-based answer
                        print(f"[RAG ENGINE] LLM not available, using extraction...")
                        answer, confidence = self._generate_answer(query, context, question_type)
                    
                    print(f"[RAG ENGINE] Answer generated. Confidence: {confidence:.2f}")
                except Exception as e:
                    print(f"[RAG ENGINE ERROR] Answer generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Final fallback
                    answer, confidence = self._generate_answer(query, context, question_type)
        
        # If answer generation returned NO_INFO, use type-specific message
        if answer == self.NO_INFO_RESPONSE:
            answer = not_found_message
        
        # Add confidence prefix for low-confidence answers
        grounded = confidence >= 0.5
        if not grounded and answer != self.NO_INFO_RESPONSE and answer != not_found_message:
            answer = self.LOW_CONFIDENCE_PREFIX + answer
        
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence=float(confidence),
            grounded=grounded,
            raw_context=context
        )
    
    def query_with_type_filter(self, query: str, doc_type: str, top_k: int = 5) -> RAGResponse:
        """Query with document type filter (e.g., only PDFs of type 'syllabus')."""
        if not self.index:
            return RAGResponse(
                query=query,
                answer="Knowledge base not initialized.",
                citations=[],
                confidence=0.0,
                grounded=False,
                raw_context=""
            )
        
        results = self.index.search_by_type(query, doc_type, top_k)
        
        if not results:
            return self.query(query, top_k)  # Fallback to general search
        
        context = self._build_context(results)
        citations = self._create_citations(results)
        question_type = self._detect_question_type(query)
        answer, confidence = self._generate_answer(query, context, question_type)
        
        grounded = confidence >= 0.5
        if not grounded and answer != self.NO_INFO_RESPONSE:
            answer = self.LOW_CONFIDENCE_PREFIX + answer
        
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence=float(confidence),  # Convert to Python float
            grounded=grounded,
            raw_context=context
        )
    
    def format_response_with_citations(self, response: RAGResponse) -> Dict:
        """Format response for API output."""
        formatted_citations = []
        for citation in response.citations:
            formatted_citations.append({
                'type': citation.source_type,
                'name': citation.source_name,
                'url': citation.source_url if citation.source_type == 'webpage' else None,
                'page': citation.page_number,
                'relevance': float(round(citation.relevance_score, 3)),  # Convert numpy.float32 to Python float
                'snippet': citation.snippet
            })
        
        return {
            'query': response.query,
            'answer': response.answer,
            'citations': formatted_citations,
            'confidence': float(round(response.confidence, 3)),  # Convert numpy.float32 to Python float
            'grounded': response.grounded,
            'sources_count': len(response.citations)
        }


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def initialize_rag_engine(knowledge_index: KnowledgeIndex) -> RAGEngine:
    """Initialize the RAG engine with a knowledge index."""
    global _rag_engine
    _rag_engine = RAGEngine(knowledge_index)
    return _rag_engine


if __name__ == "__main__":
    # Test RAG engine
    from modules.embeddings import KnowledgeIndex
    
    # Load existing index
    index = KnowledgeIndex()
    if index.load():
        engine = RAGEngine(index)
        
        # Test queries
        test_queries = [
            "What are the admission requirements?",
            "What is the fee structure?",
            "When is the semester exam?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            response = engine.query(query)
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Citations: {len(response.citations)}")
            for citation in response.citations[:2]:
                print(f"  - {citation.source_name} ({citation.relevance_score:.2f})")
    else:
        print("No index found. Run indexing first.")
