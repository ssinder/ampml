import requests
import time
from typing import List, Dict, Set
import json


class UniProtPeptideQuery:
    def __init__(self, rate_limit: float = 1.0):
        self.base_url = "https://rest.uniprot.org"
        self.rate_limit = rate_limit
        self.last_request = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def stream_results(self, query: str, format: str = "json") -> List[Dict]:
        """Stream all results without pagination"""
        self._rate_limit()

        endpoint = f"{self.base_url}/uniprotkb/stream"

        params = {
            "query": query,
            "format": format
        }

        print(f"Querying with stream endpoint: {query}")
        print("This may take a while...")

        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get('results', [])

        print(f"Total results retrieved: {len(results)}")

        return results

    def extract_keywords(self, results: List[Dict]) -> Set[str]:
        """Extract all unique keywords from results"""
        keywords = set()

        for i, result in enumerate(results):
            if 'keywords' in result:
                for keyword_data in result['keywords']:
                    if 'name' in keyword_data:
                        keywords.add(keyword_data['name'])

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(results)} entries...")

        return keywords

    def analyze_short_peptides(self, query: str = "reviewed:true AND length:[1 TO 60]") -> Dict:
        """Analyze short peptides and return statistics"""
        print("=" * 60)
        print("UniProt Short Peptide Query")
        print("=" * 60)
        print()

        print("Fetching all results using stream endpoint...")
        print("-" * 60)

        results = self.stream_results(query)

        total_count = len(results)
        print(f"\nTotal proteins found: {total_count}")

        if total_count == 0:
            return {
                "total_count": 0,
                "keywords": [],
                "keyword_count": 0
            }

        print("\nExtracting keywords...")
        print("-" * 60)
        keywords = self.extract_keywords(results)
        keyword_list = sorted(list(keywords))
        keyword_count = len(keyword_list)

        print(f"Unique keywords found: {keyword_count}")

        return {
            "total_count": total_count,
            "keywords": keyword_list,
            "keyword_count": keyword_count
        }

    def save_results(self, data: Dict, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {filename}")


def main():
    """Main function to query short peptides"""
    client = UniProtPeptideQuery(rate_limit=1.0)

    query = "reviewed:true AND length:[1 TO 60]"

    try:
        results = client.analyze_short_peptides(query)

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total short peptides (1-60 aa): {results['total_count']}")
        print(f"Unique keyword types: {results['keyword_count']}")
        print()

        if results['keywords']:
            print("Keywords found:")
            for i, keyword in enumerate(results['keywords'], 1):
                print(f"  {i}. {keyword}")

        print("\n" + "=" * 60)

        output_file = "short_peptide_results_complete.json"
        client.save_results(results, output_file)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
