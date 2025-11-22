"""Cloud-based LLM API backends - no local models needed.

Options:
1. GitHub Models API (FREE, unlimited) - ⭐ RECOMMENDED NOW
2. Groq API (fast, free tier) 
3. Hugging Face Inference API (free tier) 
4. Google Colab (100% free, GPU access)
5. Together AI (free tier)

This allows running without local GPU/CPU constraints.
"""

from typing import List, Optional
import requests
import json
import os
import time

try:
    from groq import Groq
except ImportError:
    Groq = None


class GeminiBackend:
    """Google Gemini API - FREE and powerful
    
    Free tier: 60 requests/minute (generous)
    Speed: ~1-2 seconds
    Models: gemini-1.5-flash (fast), gemini-1.5-pro (powerful)
    
    Get free API key: https://aistudio.google.com/app/apikey
    Set env var: GEMINI_API_KEY
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-lite"):
        """Initialize Gemini backend.
        
        Args:
            api_key: Gemini API key (or set env var GEMINI_API_KEY)
            model: Model to use (default: gemini-2.5-flash-lite)
        """
        # Default API key embedded in code
        default_key = "AIzaSyAbOnhQ4UOvywV4om4zYoStxMTGHRGygbY"
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or default_key
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter. Get free key at: https://aistudio.google.com/app/apikey"
            )
        
        self.model = model
        # Use v1beta for gemini-1.5-flash model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        print(f"✓ Gemini backend initialized (model: {model})")
    
    def answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini API."""
        system_instruction = (
            "You are an assistant answering questions about the book 'Home Studio Recording: The Complete Guide'. "
            "You must rely strictly on the supplied passages from the book. Do not use external "
            "knowledge or assumptions beyond the provided text. If the answer is not supported by the passages, "
            "respond with 'I don't know based on the provided book excerpts.' Keep answers concise and faithful to the context."
        )
        
        prompt = f"{system_instruction}\n\nBook context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800,
                "stopSequences": [],
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle API typo: sometimes it returns 'candiddates' instead of 'candidates'
            candidates_key = None
            if "candidates" in result:
                candidates_key = "candidates"
            elif "candiddates" in result:  # Handle API typo
                candidates_key = "candiddates"
            
            if not candidates_key:
                raise RuntimeError(f"Gemini API error: No candidates in response: {json.dumps(result)}")
            
            # Handle different response formats
            if len(result[candidates_key]) > 0:
                candidate = result[candidates_key][0]
                
                # Check finish reason
                finish_reason = candidate.get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    raise RuntimeError(f"Gemini API error: Response truncated (MAX_TOKENS). Increase maxOutputTokens or reduce input.")
                
                # Try to extract text from various possible locations
                if "content" in candidate:
                    content = candidate["content"]
                    
                    # Check for parts array
                    if "parts" in content and len(content["parts"]) > 0:
                        for part in content["parts"]:
                            if "text" in part and part["text"].strip():
                                return part["text"].strip()
                    
                    # Check for direct text field
                    if "text" in content and isinstance(content["text"], str) and content["text"].strip():
                        return content["text"].strip()
                
                # Check for text directly in candidate
                if "text" in candidate and isinstance(candidate["text"], str) and candidate["text"].strip():
                    return candidate["text"].strip()
                
                # Debug: print what we got
                raise RuntimeError(f"Gemini API error: No text found in response. Candidate structure: {json.dumps(candidate)}")
            
            raise RuntimeError(f"Gemini API error: Empty candidates array")
            
        except requests.exceptions.HTTPError as e:
            # Re-raise to allow fallback handling
            raise RuntimeError(f"Gemini API error: {e.response.status_code} - {e.response.text}")
        except KeyError as e:
            # Handle missing keys in response
            raise RuntimeError(f"Gemini API error: Unexpected response format - missing key: {e}. Response: {json.dumps(result) if 'result' in locals() else 'N/A'}")
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")


class GitHubModelsBackend:
    """GitHub Models API - Free unlimited LLM access
    
    Free tier: Unlimited requests
    Speed: <2 seconds for most queries
    Models: Claude, GPT, Llama, Phi, etc.
    
    Get free token: https://github.com/settings/tokens (create Personal Access Token)
    Or use: GitHub_TOKEN environment variable
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize GitHub Models backend.
        
        Args:
            api_key: GitHub PAT token (or set env var GITHUB_TOKEN)
            model: Model to use (default: gpt-4o-mini - better rate limits)
        """
        self.api_key = api_key or os.getenv("GITHUB_TOKEN")
        if not self.api_key:
            raise ValueError(
                "GitHub token not found. Set GITHUB_TOKEN environment variable "
                "or pass api_key parameter. Create token at: https://github.com/settings/tokens"
            )
        
        self.model = model
        self.base_url = "https://models.inference.ai.azure.com"
        print(f"✓ GitHub Models backend initialized (model: {model})")
    
    def answer(self, question: str, context: str) -> str:
        """Generate answer using GitHub Models API with retry logic."""
        system_prompt = (
            "You are an assistant answering questions about the book 'Home Studio Recording: The Complete Guide'. "
            "You must rely strictly on the supplied passages from the book. Do not use the internet, external "
            "knowledge, or assumptions beyond the provided text. If the answer is not supported by the passages, "
            "respond with 'I don't know based on the provided book excerpts.' Keep answers concise and faithful to the context."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Book context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."},
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.3,
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s
                        wait_time = 2 ** (attempt + 1)
                        print(f"Rate limit hit, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final attempt failed - return context directly
                        return (
                            "⚠️ API rate limit reached. Please wait a moment and try again. "
                            "Here are the most relevant passages I found:\n\n"
                            f"{context[:800]}"
                        )
                raise RuntimeError(f"GitHub Models API error: {e}")
            except Exception as e:
                raise RuntimeError(f"GitHub Models API error: {e}")



class HuggingFaceInferenceBackend:
    """Hugging Face Inference API
    
    Free tier: Limited requests
    Speed: ~2-3 seconds
    Models: Multiple options (Mistral, Llama, etc.)
    
    Get free API key: https://huggingface.co/settings/tokens
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize HF Inference backend."""
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key not found. Set HUGGINGFACE_API_KEY environment variable "
                "or pass api_key parameter. Get free key: https://huggingface.co/settings/tokens"
            )
        
        self.model = model
        self.base_url = f"https://api-inference.huggingface.co/models/{model}"
        print(f"✓ HuggingFace Inference backend initialized (model: {model})")
    
    def answer(self, question: str, context: str) -> str:
        """Generate answer using HF Inference API."""
        prompt = f"""You are an assistant answering questions about 'Home Studio Recording: The Complete Guide'.
    Use ONLY the provided context and do not rely on outside knowledge, the internet, or assumptions.
    If the answer is not supported by the context, reply with "I don't know based on the provided book excerpts."

Context:
{context}

Question: {question}

Answer:"""
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.3,
            }
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return str(result).strip()
        except Exception as e:
            raise RuntimeError(f"HuggingFace API error: {e}")


class ColabBackend:
    """Google Colab Backend (100% FREE, no API key needed!)
    
    Setup:
    1. Open: https://colab.research.google.com
    2. Create new notebook
    3. Run setup_colab_server.py (provided separately)
    4. Share the ngrok URL with VS Code
    5. Connect using this backend
    
    Advantages:
    - Completely FREE (Google provides GPU/TPU)
    - No API key needed
    - Access to powerful GPUs
    - Perfect for testing
    """
    
    def __init__(self, server_url: str):
        """Initialize Colab backend.
        
        Args:
            server_url: ngrok URL from Colab (e.g., https://xxx.ngrok.io)
        """
        self.server_url = server_url.rstrip("/")
        print(f"✓ Colab backend initialized (server: {server_url})")
    
    def answer(self, question: str, context: str) -> str:
        """Generate answer using Colab server."""
        payload = {
            "question": question,
            "context": context,
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/answer",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["answer"]
        except Exception as e:
            raise RuntimeError(f"Colab API error: {e}")


class TogetherAIBackend:
    """Together AI - Free tier with good models
    
    Free tier: API credits
    Speed: <2 seconds
    Models: Multiple open-source models
    
    Get free API key: https://www.together.ai
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize Together AI backend."""
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Together AI API key not found. Set TOGETHER_API_KEY environment variable. "
                "Get free key: https://www.together.ai"
            )
        
        self.model = model
        self.base_url = "https://api.together.xyz/inference"
        print(f"✓ Together AI backend initialized (model: {model})")
    
    def answer(self, question: str, context: str) -> str:
        """Generate answer using Together AI API."""
        prompt = f"""You are an assistant answering questions about 'Home Studio Recording: The Complete Guide'.
    Use ONLY the provided context and do not use the internet or external knowledge.
    If the information is not in the context, respond with "I don't know based on the provided book excerpts."

Context: {context}

Question: {question}

Answer:"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.3,
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["output"]["choices"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Together AI error: {e}")


def get_api_backend(backend_name: str, **kwargs):
    """Factory function to get API backend.
    
    Args:
        backend_name: 'gemini' (recommended), 'github', 'huggingface', 'together', or 'colab'
        **kwargs: Backend-specific arguments
    
    Returns:
        Backend instance
    """
    backends = {
        "gemini": GeminiBackend,
        "github": GitHubModelsBackend,
        "huggingface": HuggingFaceInferenceBackend,
        "together": TogetherAIBackend,
        "colab": ColabBackend,
    }
    
    backend_class = backends.get(backend_name.lower())
    if not backend_class:
        raise ValueError(f"Unknown backend: {backend_name}. Options: {list(backends.keys())}")
    
    return backend_class(**kwargs)
