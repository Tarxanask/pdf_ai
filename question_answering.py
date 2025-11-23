"""Question answering backend over processed_data.json.

Pipeline:
1. Load processed_data.json via DatasetLoader
2. Turn book paragraphs (and chapter titles) into embedding vectors using sentence-transformers
3. For each user question:
   - normalise + extract keywords
   - embed the question
   - retrieve top-k relevant passages by vector similarity (+ simple keyword scoring)
   - call LLM (Cloud API: GitHub Models, Groq, HuggingFace, Colab) with question and context

Supports Cloud LLM APIs (all free):
- GitHub Models API (UNLIMITED - recommended)
- Groq API (fast, 30 req/min)
- HuggingFace Inference (free tier)
- Google Colab (100% free GPU access)
- Together AI (free tier)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from dataset_loader import DatasetLoader
from gemini_embeddings import GeminiEmbeddingService


# --------------------------- Constants ---------------------------

# Page offset: book page 1 = PDF page 13
PDF_PAGE_OFFSET = 12

# --------------------------- Data structures ---------------------------

@dataclass
class Passage:
    paragraph_id: int
    page_number: int
    text: str
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None


@dataclass
class RetrievedPassage:
    passage: Passage
    score: float


# --------------------------- Embedding helper ---------------------------
# Note: Embedding now uses Gemini API (gemini_embeddings.py)
# This eliminates the 80-120MB local model memory footprint


# --------------------------- Retrieval index ---------------------------

class BookIndex:
    """In-memory embedding index built from processed_data.json paragraphs."""

    def __init__(self, data_dir: str = ".", max_paragraphs: Optional[int] = None):
        self.loader = DatasetLoader(data_dir=data_dir)
        self.passages: List[Passage] = []
        self.chapters: List[Dict[str, Any]] = []  # Store chapter metadata
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_backend = None

        self._build_passages(max_paragraphs=max_paragraphs)

    # ----------------- building -----------------

    def _build_passages(self, max_paragraphs: Optional[int] = None) -> None:
        data = self.loader.load_json()
        paragraphs = data.get("paragraphs", [])
        chapters = data.get("chapters", [])
        
        # Store chapters for TOC generation
        self.chapters = chapters

        # quick lookup: page_number -> (chapter_number, chapter_title)
        page_to_chapter: Dict[int, Tuple[Optional[int], Optional[str]]] = {}
        for ch in chapters:
            chapter_number = self._to_optional_int(ch.get("chapter_number"))
            raw_title = ch.get("title")
            clean_title = self._normalise_chapter_title(raw_title)

            try:
                start = int(ch.get("page_start", 0))
            except (TypeError, ValueError):
                start = 0
            try:
                end = int(ch.get("page_end", start))
            except (TypeError, ValueError):
                end = start

            for page in range(start, end + 1):
                page_to_chapter[page] = (chapter_number, clean_title)

        limit = max_paragraphs or len(paragraphs)
        for entry in paragraphs[:limit]:
            paragraph_id = self._to_optional_int(entry.get("paragraph_id"))
            page_number = self._to_optional_int(entry.get("page_number"))
            if paragraph_id is None or page_number is None:
                continue

            chapter_meta = page_to_chapter.get(page_number, (None, None))
            chapter_number, chapter_title = chapter_meta
            cleaned_text = self._clean_paragraph_text(entry.get("text"))
            if not cleaned_text:
                continue

            self.passages.append(
                Passage(
                    paragraph_id=paragraph_id,
                    page_number=page_number,
                    text=cleaned_text,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                )
            )

    @staticmethod
    def _to_optional_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalise_chapter_title(raw_title: Any) -> Optional[str]:
        if not isinstance(raw_title, str):
            raw_title = ""

        title = raw_title.strip()
        if not title:
            return None

        title = re.sub(r"\s*\d+\s*$", "", title).strip()
        if not title:
            return None

        tokens = title.split()
        if tokens and all(len(token) == 1 for token in tokens):
            title = "".join(tokens)
        else:
            title = " ".join(tokens)

        return title or None

    @classmethod
    def _clean_paragraph_text(cls, text: Any) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = text.strip()
        if not cleaned:
            return ""

        header_patterns = [
            re.compile(
                r"^Home\s+Studio\s+Recording(?::?\s+The\s+Complete\s+Guide)?\s*",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^Warren\s+Huart\s+and\s+Jerry\s+Hammack\s*",
                flags=re.IGNORECASE,
            ),
        ]

        for pattern in header_patterns:
            while True:
                new_value = pattern.sub("", cleaned, count=1)
                if new_value == cleaned:
                    break
                cleaned = new_value.lstrip()

        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def _ensure_embeddings(self) -> None:
        """Lazy-load embeddings using Gemini API (on first query only)."""
        if self.embeddings is not None:
            return
        
        # Don't generate all embeddings at once - too slow for free tier
        # Instead, we'll generate them on-demand in smaller batches
        print(f"⏳ Initializing embedding service for {len(self.passages)} passages...")
        print("   Embeddings will be generated on-demand for faster startup.")
        
        self.embedding_backend = GeminiEmbeddingService()
        
        # Initialize empty embeddings - we'll fill them gradually
        self.embeddings = None
        self.embedded_count = 0
        
        print(f"✓ Embedding service ready (on-demand mode)")

    def _ensure_embeddings_ready(self):
        """Generate embeddings on-demand - much faster startup for free tier."""
        if self.embeddings is not None:
            return  # Already have embeddings
        
        print(f"⏳ Generating embeddings for {len(self.passages)} passages...")
        print("   This may take 2-3 minutes but only happens once.")
        
        texts = [p.text for p in self.passages]
        self.embeddings = self.embedding_backend.embed_texts(texts, task_type="retrieval_document")
        
        print(f"✓ All embeddings cached ({self.embeddings.shape})")

    # ----------------- retrieval -----------------

    @staticmethod
    def _detect_page_query(question: str) -> Optional[int]:
        """Detect if question is asking about a specific page number.
        
        Handles both book pages and PDF pages.
        Examples: 'page 14', 'page 145', 'the 12th page', 'p. 23'
        """
        q = question.lower()
        # Match patterns like: "page 14", "page 145", "p. 23", "p14"
        patterns = [
            r"\bpage\s+(\d+)",
            r"\bp\.?\s*(\d+)",
            r"(\d+)(?:th|st|nd|rd)\s+page",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_keywords(question: str, max_tokens: int = 10) -> List[str]:
        """Very simple keyword extractor: lowercase, drop stopwords, keep alphanumerics."""
        q = question.lower()
        tokens = re.findall(r"[a-zA-Z0-9]+", q)
        stop = {
            "the", "and", "or", "a", "an", "of", "for", "in", "on", "to", "is", "are", "be",
            "this", "that", "it", "as", "with", "by", "from", "at", "we", "you", "i",
        }
        keywords = [t for t in tokens if t not in stop]
        # unique, preserve order
        seen = set()
        dedup: List[str] = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                dedup.append(k)
        return dedup[:max_tokens]

    def _is_chapter_overview_query(self, question: str) -> bool:
        """Detect if question is asking for chapter summaries or book structure."""
        q = question.lower()
        patterns = [
            r"\ball\s+chapters",
            r"\bchapter\s+descriptions?",
            r"\bchapter\s+summaries",
            r"\bbook\s+structure",
            r"\bhow\s+is\s+(?:the\s+)?book\s+(?:organized|formed|structured)",
            r"\btable\s+of\s+contents",
            r"\bwhat\s+(?:is\s+)?in\s+(?:the\s+)?(?:beginning|middle|end)",
            r"\bparts?\s+of\s+(?:the\s+)?book",
            r"\bshort\s+description\s+of\s+chapters",
        ]
        for pattern in patterns:
            if re.search(pattern, q):
                return True
        return False

    def _build_chapter_overview(self) -> str:
        """Build a structured overview of all chapters."""
        if not self.chapters:
            return "No chapter information available."
        
        overview = "Book Structure Overview:\n\n"
        for ch in self.chapters:
            chapter_num = ch.get("chapter_number", "")
            title = ch.get("title", "Untitled")
            page_start = ch.get("page_start", "?")
            page_end = ch.get("page_end", "?")
            
            # Clean up title
            clean_title = self._normalise_chapter_title(title) or title
            
            if chapter_num:
                overview += f"Chapter {chapter_num}: {clean_title}\n"
            else:
                overview += f"{clean_title}\n"
            overview += f"  Pages: {page_start}-{page_end}\n\n"
        
        return overview.strip()
    
    def _is_general_book_query(self, question: str) -> bool:
        """Detect queries asking for general book overview/summary."""
        q = question.lower()
        patterns = [
            r"what\s+(?:is\s+)?(?:this\s+)?(?:book|file)\s+about",
            r"(?:general|brief|quick)\s+(?:overview|summary|description)",
            r"tell\s+me\s+about\s+(?:this\s+)?(?:book|file)",
            r"what\s+(?:does\s+)?(?:this\s+)?(?:book|file)\s+cover",
            r"(?:main|key)\s+(?:topics|themes|subjects)",
            r"(?:book|file)\s+(?:summary|overview|description)",
            r"what\s+kind\s+of\s+book",
            r"purpose\s+of\s+(?:this\s+)?book",
            r"in\s+general\s+what",
            r"about\s+the\s+book",
        ]
        for pattern in patterns:
            if re.search(pattern, q):
                return True
        return False
    
    def _get_general_info_passages(self) -> List[Passage]:
        """Return high-confidence passages that introduce the book."""
        preferred_ids = {16, 17, 18}
        general_passages: List[Passage] = []
        for passage in self.passages:
            if passage.paragraph_id in preferred_ids:
                general_passages.append(passage)

        if general_passages:
            general_passages.sort(key=lambda p: p.paragraph_id)
            return general_passages

        # Fallback to earliest pages if specific intro paragraphs not found
        return self.passages[: min(3, len(self.passages))]

    def _build_book_summary(self) -> str:
        """Build a comprehensive summary of the book based on chapter titles and content."""
        if not self.chapters:
            return "This appears to be a comprehensive guide to music production and audio engineering."
        
        # Get first few chapters to understand book structure
        intro_chapters = self.chapters[:5] if len(self.chapters) >= 5 else self.chapters
        
        summary = [
            "This book is a comprehensive guide to music production and audio engineering for home studios. ",
            "It covers professional recording, mixing, and mastering techniques.\n\n",
            "The book is organized into the following sections:\n"
        ]
        
        # Add all chapters with brief description
        for ch in self.chapters:
            ch_num = ch.get("chapter_number", "")
            title = self._normalise_chapter_title(ch.get("title", ""))
            if ch_num and title:
                summary.append(f"• Chapter {ch_num}: {title}")
        
        summary.append("\n\nKey topics include:")
        summary.append("- Studio setup (digital, analogue, and hybrid systems)")
        summary.append("- Microphone selection and placement techniques")
        summary.append("- Recording vocals, instruments, and live performances")
        summary.append("- Mixing fundamentals and advanced techniques")
        summary.append("- Mastering for different formats (CD, vinyl, streaming)")
        summary.append("- Building relationships with artists")
        summary.append("- Professional workflow and session management")
        summary.append("\nThis book is designed for home studio engineers, producers, and anyone ")
        summary.append("looking to improve their music production skills with practical, ")
        summary.append("professional-level guidance.")
        
        return "\n".join(summary)

    def retrieve(self, question: str, top_k: int = 8) -> List[RetrievedPassage]:
        """Retrieve top-k relevant passages by cosine similarity + keyword overlap.
        
        Handles special cases:
        - General book queries: returns comprehensive book summary
        - Page-specific queries: searches ±5 pages around target
        - Chapter overview queries: returns chapter structure information
        """
        if not question.strip():
            return []

        # Handle general book overview FIRST so it works without embeddings
        if self._is_general_book_query(question):
            general_passages = self._get_general_info_passages()
            retrieved: List[RetrievedPassage] = []
            for idx, p in enumerate(general_passages):
                score = 1.0 - (idx * 0.05)
                retrieved.append(RetrievedPassage(passage=p, score=score))
            return retrieved

        # Check for chapter overview query
        if self._is_chapter_overview_query(question):
            # Return the actual Table of Contents from the book (paragraph 4, page 5)
            toc_passage = None
            for p in self.passages:
                if p.paragraph_id == 4 and 'table of contents' in p.text.lower():
                    toc_passage = p
                    break
            
            if toc_passage:
                return [RetrievedPassage(passage=toc_passage, score=1.0)]
            else:
                # Fallback: build synthetic TOC if the actual one isn't found
                overview_text = self._build_chapter_overview()
                special_passage = Passage(
                    paragraph_id=-1,
                    page_number=1,
                    text=overview_text,
                    chapter_number=None,
                    chapter_title="Table of Contents"
                )
                return [RetrievedPassage(passage=special_passage, score=1.0)]

        # Ensure embeddings only when needed for page or semantic search
        self._ensure_embeddings()
        assert self.embeddings is not None and self.embedding_backend is not None

        # Check for page-specific query
        target_page = self._detect_page_query(question)
        if target_page:
            # Convert book page to actual page (accounting for PDF offset)
            # If user says "page 1" they likely mean book page 1 (PDF page 13)
            # If they say "page 145" they likely mean PDF page 145
            # Use heuristic: if page > 50, treat as PDF page, else as book page
            if target_page <= 50:
                actual_page = target_page + PDF_PAGE_OFFSET
            else:
                actual_page = target_page
            
            # Search ±5 pages around target
            page_passages = [
                (idx, p) for idx, p in enumerate(self.passages)
                if abs(p.page_number - actual_page) <= 5 and len(p.text) > 50  # Filter out empty pages
            ]
            
            if page_passages:
                # Sort by proximity to target page
                page_passages.sort(key=lambda x: abs(x[1].page_number - actual_page))
                # Return top_k results from nearby pages
                results = []
                for idx, p in page_passages[:top_k]:
                    # Calculate score based on page proximity and content length
                    distance = abs(p.page_number - actual_page)
                    proximity_score = 1.0 - (distance / 6.0)  # 0.0-1.0 score
                    # Boost score for longer, more substantive content
                    content_boost = min(len(p.text) / 500.0, 0.2)
                    results.append(RetrievedPassage(passage=p, score=proximity_score + content_boost))
                return results
            else:
                # No substantive content found, create helpful message
                fallback_passage = Passage(
                    paragraph_id=-2,
                    page_number=actual_page,
                    text=f"Page {target_page} appears to contain primarily chapter headers or minimal content. Try asking about the chapter content or a nearby page.",
                    chapter_number=None,
                    chapter_title=None
                )
                return [RetrievedPassage(passage=fallback_passage, score=0.5)]

        # Ensure we have embeddings (generate on-demand if needed)
        self._ensure_embeddings_ready()

        # Standard semantic search
        q_emb = self.embedding_backend.embed_query(question)
        # cosine because everything is unit-normalised
        sims = self.embeddings @ q_emb

        keywords = self._extract_keywords(question)
        kw_set = set(keywords)

        scores: List[Tuple[float, int]] = []  # (score, index)
        for idx, passage in enumerate(self.passages):
            base = float(sims[idx])

            # simple keyword overlap bonus
            text_tokens = set(re.findall(r"[a-zA-Z0-9]+", passage.text.lower()))
            overlap = len(text_tokens & kw_set)
            bonus = 0.05 * overlap

            scores.append((base + bonus, idx))

        scores.sort(reverse=True, key=lambda x: x[0])
        out: List[RetrievedPassage] = []
        for score, idx in scores[:top_k]:
            out.append(RetrievedPassage(passage=self.passages[idx], score=score))
        return out


# --------------------------- LLM answering ---------------------------

class LLMBackend:
    """LLM inference using Cloud APIs with fallback support and caching."""

    def __init__(
        self,
        backend_type: str = "gemini",  # "gemini" (default), "github", "groq", "huggingface", "together", "colab"
        api_key: str = None,
        server_url: str = None,
        cache_file: str = None,
    ):
        """Initialize LLM backend with fallback support.
        
        Args:
            backend_type: "gemini" (default, fast & free), "github" (fallback), "groq", "huggingface", "together", or "colab"
            api_key: API key for backends (or use env vars: GEMINI_API_KEY, GITHUB_TOKEN, etc)
            server_url: Server URL (for Colab backend)
            cache_file: Path to cache file for storing previous Q&A (qa_history.jsonl)
        """
        self.backend_type = backend_type.lower()
        self.last_context: str = ""
        self.last_passages: List[Dict[str, Any]] = []
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_questions: Dict[str, str] = {}  # Store original questions for fuzzy matching
        
        # Load cache if available
        if self.cache_file:
            self._load_cache()
        
        # Load API backend
        try:
            from api_backends import get_api_backend
        except ImportError:
            raise ImportError(
                "api_backends.py not found. Please ensure it's in the same directory. "
                "This module provides free Cloud LLM APIs."
            )
        
        # Initialize primary backend
        if self.backend_type == "colab":
            if not server_url:
                raise ValueError("Colab backend requires server_url parameter")
            self.backend = get_api_backend("colab", server_url=server_url)
        else:
            try:
                self.backend = get_api_backend(self.backend_type, api_key=api_key)
            except Exception as e:
                print(f"⚠️ Failed to initialize {self.backend_type} backend: {e}")
                # Try fallback to GitHub if Gemini fails
                if self.backend_type == "gemini":
                    print("Falling back to GitHub Models backend...")
                    self.backend_type = "github"
                    self.backend = get_api_backend("github", api_key=api_key)
                else:
                    raise
        
        # Initialize fallback backend (GitHub) if primary is Gemini
        self.fallback_backend = None
        if self.backend_type == "gemini":
            try:
                import os
                self.fallback_backend = get_api_backend("github", api_key=os.getenv("GITHUB_TOKEN"))
                print("✓ GitHub Models fallback backend ready")
            except Exception as e:
                print(f"⚠️ GitHub fallback not available: {e}")

    def _load_cache(self) -> None:
        """Load cache from qa_history.jsonl file."""
        try:
            if not os.path.exists(self.cache_file):
                return
            
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        question = record.get("question", "")
                        answer = record.get("answer", "")
                        
                        # Skip malformed answers
                        if answer == "{'role': 'model'}" or len(answer) < 20:
                            continue
                        
                        # Create cache key from normalized question only (ignore context)
                        normalized_q = self._normalize_question(question)
                        cache_key = normalized_q
                        self.cache[cache_key] = {
                            "answer": answer,
                            "context_passages": record.get("context_passages", []),
                            "original_question": question
                        }
                        # Store original question for display
                        self.cache_questions[cache_key] = question
                    except json.JSONDecodeError:
                        continue
            
            print(f"✓ Loaded {len(self.cache)} cached answers from {self.cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
    
    @staticmethod
    def _normalize_question(question: str) -> str:
        """Normalize question for very conservative fuzzy matching."""
        # Lowercase
        q = question.lower().strip()
        # Remove common punctuation at end
        q = re.sub(r'[?.!,;:]+$', '', q)
        # Normalize whitespace
        q = ' '.join(q.split())
        # Remove common articles and filler words that don't change meaning
        q = re.sub(r'\b(the|a|an)\b', '', q)
        q = ' '.join(q.split())  # Clean up extra spaces again
        return q
    
    def _get_cached_answer(self, question: str, context: str) -> Optional[str]:
        """Check if we have a cached answer for this question (ignoring context)."""
        if not self.cache:
            return None
        
        # Normalize the question and use as cache key
        normalized_question = self._normalize_question(question)
        
        if normalized_question in self.cache:
            cached_data = self.cache[normalized_question]
            original_q = cached_data.get("original_question", "")
            print(f"✓ Using cached answer (matched: '{original_q}')")
            self.last_passages = cached_data.get("context_passages", [])
            return cached_data["answer"]
        
        return None

    def answer(self, question: str, passages: List["RetrievedPassage"]) -> str:
        """Generate an answer based on question and retrieved passages with caching and fallback."""
        if not passages:
            return "I don't have any context from the book to answer this question."

        context_chunks = []
        passage_records: List[Dict[str, Any]] = []
        for rp in passages:
            p = rp.passage
            chapter_label = p.chapter_title
            if not chapter_label and p.chapter_number is not None:
                chapter_label = f"Chapter {p.chapter_number}"
            header = f"[p{p.paragraph_id}, page {p.page_number}, chapter={chapter_label or 'N/A'}]"
            context_chunks.append(header + "\n" + p.text)
            passage_records.append(
                {
                    "paragraph_id": p.paragraph_id,
                    "page_number": p.page_number,
                    "chapter_number": p.chapter_number,
                    "chapter_title": p.chapter_title,
                    "section_title": p.section_title,
                    "passage_text": p.text,
                    "similarity_score": float(rp.score),
                }
            )

        context = "\n\n".join(context_chunks)
        self.last_context = context
        self.last_passages = passage_records
        
        # Check cache first
        cached_answer = self._get_cached_answer(question, context)
        if cached_answer:
            return cached_answer
        
        # Try primary backend (Gemini)
        try:
            return self.backend.answer(question, context)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit or quota error
            is_limit_error = any(term in error_str for term in [
                "rate limit", "quota", "429", "resource_exhausted",
                "too many requests", "limit exceeded"
            ])
            
            # Try fallback if available and error is recoverable
            if self.fallback_backend and (is_limit_error or "api" in error_str):
                print(f"⚠️ Primary backend error: {e}")
                print("→ Using GitHub Models fallback...")
                try:
                    return self.fallback_backend.answer(question, context)
                except Exception as fallback_error:
                    print(f"⚠️ Fallback also failed: {fallback_error}")
                    return f"Error: Both primary and fallback backends failed. Please try again later.\n\nContext preview:\n{context[:500]}"
            else:
                # Re-raise if no fallback or unrecoverable error
                raise


# --------------------------- High-level API ---------------------------

class BookQAService:
    """High-level service: given a question, return answer + supporting passages."""

    def __init__(
        self,
        data_dir: str = ".",
        max_paragraphs: Optional[int] = None,
        backend_type: str = "gemini",
        api_key: str = None,
        server_url: str = None,
        history_path: Optional[str] = "qa_history.jsonl",
    ):
        """Initialize QA service.
        
        Args:
            data_dir: Directory with processed_data.json
            max_paragraphs: Max paragraphs to load (for testing)
            backend_type: "gemini" (default, fast & free), "github" (fallback), "groq", "huggingface", "together", or "colab"
            api_key: API key for Cloud LLMs (or use env vars: GEMINI_API_KEY, GITHUB_TOKEN, etc)
            server_url: Server URL for Colab backend
            history_path: Path to append Q&A history as JSON lines (set to None to disable, also used for caching)
        """
        self.index = BookIndex(data_dir=data_dir, max_paragraphs=max_paragraphs)
        self.llm = LLMBackend(
            backend_type=backend_type,
            api_key=api_key,
            server_url=server_url,
            cache_file=history_path,
        )
        self.history_path = history_path

    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer a question using semantic search and LLM.
        
        Simple approach: retrieve top_k relevant passages and generate answer.
        """
        # Retrieve relevant passages
        retrieved = self.index.retrieve(question, top_k=top_k)
        
        # Generate answer using LLM
        answer = self.llm.answer(question, retrieved)
        
        # Build context passages for display
        context_passages = [
            {
                "paragraph_id": rp.passage.paragraph_id,
                "page_number": rp.passage.page_number,
                "chapter_number": rp.passage.chapter_number,
                "chapter_title": rp.passage.chapter_title,
                "section_title": getattr(rp.passage, 'section_title', None),
                "passage_text": rp.passage.text,
                "similarity_score": rp.score,
            }
            for rp in retrieved
        ]
        
        # Build passages list for compatibility
        passages_for_result = [
            {
                "paragraph_id": rp.passage.paragraph_id,
                "page_number": rp.passage.page_number,
                "chapter_number": rp.passage.chapter_number,
                "chapter_title": rp.passage.chapter_title,
                "score": rp.score,
                "text": rp.passage.text,
            }
            for rp in retrieved
        ]
        
        # Combine context text
        combined_context = "\n\n---\n\n".join([rp.passage.text for rp in retrieved])
        
        result = {
            "question": question,
            "answer": answer,
            "passages": passages_for_result,
            "context_passages": context_passages,
            "combined_context": combined_context,
        }
        
        self._log_interaction(question, answer, retrieved, top_k)
        return result

    def _log_interaction(
        self,
        question: str,
        answer: str,
        retrieved: List[RetrievedPassage],
        top_k: int,
    ) -> None:
        """Append question/answer/passages to history log if enabled."""
        if not self.history_path:
            return

        if self.llm.last_passages:
            context_passages = [
                {
                    "paragraph_id": entry["paragraph_id"],
                    "page_number": entry["page_number"],
                    "chapter_number": entry.get("chapter_number"),
                    "chapter_title": entry["chapter_title"],
                    "section_title": entry["section_title"],
                    "passage_text": entry["passage_text"],
                    "similarity_score": round(entry["similarity_score"], 6),
                }
                for entry in self.llm.last_passages
            ]
            combined_context = self.llm.last_context
        else:
            context_passages = [
                {
                    "paragraph_id": rp.passage.paragraph_id,
                    "page_number": rp.passage.page_number,
                    "chapter_number": rp.passage.chapter_number,
                    "chapter_title": rp.passage.chapter_title,
                    "section_title": rp.passage.section_title,
                    "passage_text": rp.passage.text,
                    "similarity_score": round(float(rp.score), 6),
                }
                for rp in retrieved
            ]
            combined_context = "\n\n".join(p["passage_text"] for p in context_passages)

        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "question": question,
            "answer": answer,
            "backend": getattr(self.llm, "backend_type", "unknown"),
            "retrieval_top_k": top_k,
            "context_passages": context_passages,
            "combined_context": combined_context,
            "source_data": "processed_data.json",
        }

        try:
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Logging errors should not break primary flow.
            print(f"[WARN] Failed to write history log: {exc}")


if __name__ == "__main__":
    import json

    qa = BookQAService(data_dir=".")
    sample_q = "What are the main advantages of modern home studios compared to traditional commercial studios?"
    result = qa.answer_question(sample_q, top_k=6)
    print(json.dumps(result, indent=2, ensure_ascii=False))
