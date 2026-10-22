import requests
import time
from typing import List, Dict, Set
import json
from collections import Counter


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

    def search_proteins(self, query: str, fields: List[str] = None,
                       size: int = 500, format: str = "json") -> Dict:
        """Search UniProt database with pagination"""
        self._rate_limit()

        endpoint = f"{self.base_url}/uniprotkb/search"

        params = {
            "query": query,
            "format": format,
            "size": size
        }

        if fields:
            params["fields"] = ",".join(fields)

        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        if format == "json":
            return response.json()
        else:
            return response.text

    def get_all_results(self, query: str, fields: List[str] = None) -> List[Dict]:
        """Get all results with pagination"""
        all_results = []
        url = f"{self.base_url}/uniprotkb/search"

        params = {
            "query": query,
            "format": "json",
            "size": 500
        }

        if fields:
            params["fields"] = ",".join(fields)

        page_count = 0
        while True:
            self._rate_limit()

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            all_results.extend(results)
            page_count += 1

            print(f"  Fetched page {page_count}: {len(results)} results (total: {len(all_results)})")

            if 'next' in data:
                url = data['next']
            else:
                break

        return all_results

    def extract_keywords(self, results: List[Dict]) -> Set[str]:
        """Extract all unique keywords from results"""
        keywords = set()

        for result in results:
            if 'keywords' in result:
                for keyword_data in result['keywords']:
                    if 'name' in keyword_data:
                        keywords.add(keyword_data['name'])

        return keywords

    def analyze_short_peptides(self, query: str = "reviewed:true AND length:[1 TO 60]") -> Dict:
        """Analyze short peptides and return statistics"""
        print(f"Querying UniProt for: {query}")
        print("-" * 60)

        fields = ["accession", "protein_name", "gene_names", "organism_name",
                  "length", "keyword", "sequence"]

        print("Fetching results...")
        results = self.get_all_results(query, fields=fields)

        total_count = len(results)
        print(f"\nTotal proteins found: {total_count}")

        if total_count == 0:
            return {
                "total_count": 0,
                "keywords": [],
                "keyword_count": 0
            }

        print("\nExtracting keywords...")
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
    print("=" * 60)
    print("UniProt Short Peptide Query")
    print("=" * 60)
    print()

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

        output_file = "short_peptide_results.json"
        client.save_results(results, output_file)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
