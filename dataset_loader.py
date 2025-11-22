"""
Dataset Loader
Easy interface to load and work with the processed datasets.
No need to parse the PDF anymore - just load the datasets!
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


class DatasetLoader:
    """Load and work with processed PDF datasets."""
    
    def __init__(self, data_dir: str = ".", json_file: str = "processed_data.json"):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing datasets
            json_file: Name of the JSON file with all data
        """
        self.data_dir = Path(data_dir)
        self.json_file = self.data_dir / json_file
        self.data = None
        
    def load_json(self) -> Dict:
        """
        Load the complete JSON dataset.
        
        Returns:
            Complete processed data dictionary
        """
        if self.data is None:
            print(f"Loading dataset from {self.json_file}...")
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print("Dataset loaded successfully!")
        return self.data
    
    def get_chapters(self) -> List[Dict]:
        """Get all chapters."""
        data = self.load_json()
        return data.get('chapters', [])
    
    def get_chapter(self, chapter_number: int) -> Optional[Dict]:
        """Get a specific chapter by number."""
        chapters = self.get_chapters()
        for ch in chapters:
            if ch.get('chapter_number') == chapter_number:
                return ch
        return None
    
    def get_tables(self) -> List[Dict]:
        """Get all tables."""
        data = self.load_json()
        return data.get('tables', [])
    
    def get_table(self, table_id: int) -> Optional[Dict]:
        """Get a specific table by ID."""
        tables = self.get_tables()
        for table in tables:
            if table.get('table_id') == table_id:
                return table
        return None
    
    def get_paragraphs(self, page_number: Optional[int] = None) -> List[Dict]:
        """
        Get all paragraphs, optionally filtered by page.
        
        Args:
            page_number: Optional page number to filter by
            
        Returns:
            List of paragraphs
        """
        data = self.load_json()
        paragraphs = data.get('paragraphs', [])
        
        if page_number:
            return [p for p in paragraphs if p.get('page_number') == page_number]
        return paragraphs
    
    def get_key_terms(self, top_n: Optional[int] = None) -> List[Dict]:
        """
        Get key terms, optionally limited to top N.
        
        Args:
            top_n: Optional limit to top N terms
            
        Returns:
            List of key terms
        """
        data = self.load_json()
        terms = data.get('key_terms', [])
        
        if top_n:
            return terms[:top_n]
        return terms
    
    def search_text(self, query: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Search for text in paragraphs.
        
        Args:
            query: Search query
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching paragraphs
        """
        paragraphs = self.get_paragraphs()
        results = []
        
        if not case_sensitive:
            query = query.lower()
        
        for para in paragraphs:
            text = para.get('text', '')
            if not case_sensitive:
                text = text.lower()
            
            if query in text:
                results.append(para)
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        data = self.load_json()
        return data.get('summary', {})
    
    def get_metadata(self) -> Dict:
        """Get PDF metadata."""
        data = self.load_json()
        return data.get('metadata', {})
    
    def load_csv(self, filename: str) -> List[Dict]:
        """
        Load a CSV dataset file.
        
        Args:
            filename: Name of CSV file in datasets directory
            
        Returns:
            List of dictionaries from CSV
        """
        csv_file = self.data_dir / "datasets" / filename
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        return data


# Convenience functions
def load_datasets(data_dir: str = ".") -> DatasetLoader:
    """
    Quick function to load datasets.
    
    Args:
        data_dir: Directory containing datasets
        
    Returns:
        DatasetLoader instance
    """
    return DatasetLoader(data_dir)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Dataset Loader - Example Usage")
    print("=" * 60)
    
    loader = load_datasets()
    
    # Get summary
    summary = loader.get_summary()
    print(f"\nDataset Summary:")
    print(f"  Chapters: {summary.get('total_chapters', 0)}")
    print(f"  Tables: {summary.get('total_tables', 0)}")
    print(f"  Paragraphs: {summary.get('total_paragraphs', 0)}")
    print(f"  Key Terms: {summary.get('total_key_terms', 0)}")
    
    # Get first chapter
    chapters = loader.get_chapters()
    if chapters:
        print(f"\nFirst Chapter: {chapters[0].get('title', 'N/A')}")
        print(f"  Pages: {chapters[0].get('page_start')} - {chapters[0].get('page_end')}")
    
    # Get top 10 key terms
    print(f"\nTop 10 Key Terms:")
    top_terms = loader.get_key_terms(top_n=10)
    for term in top_terms:
        print(f"  {term['term']}: {term['frequency']} occurrences")
    
    # Search example
    print(f"\nSearching for 'recording'...")
    results = loader.search_text("recording", case_sensitive=False)
    print(f"  Found {len(results)} paragraphs containing 'recording'")
    if results:
        print(f"  First result (first 100 chars): {results[0]['text'][:100]}...")
    
    print("\n" + "=" * 60)
    print("You can now work with datasets instead of the PDF!")
    print("=" * 60)

