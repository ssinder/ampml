import requests
import time
from typing import List, Dict, Set, Tuple
import json
from collections import Counter


class UniProtGOQuery:
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

    def extract_go_terms(self, results: List[Dict]) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, int]]:
        """Extract GO terms from results and categorize them"""
        go_terms = set()
        go_by_category = {
            'Molecular Function': [],
            'Biological Process': [],
            'Cellular Component': []
        }
        go_count = Counter()

        for i, result in enumerate(results):
            if 'uniProtKBCrossReferences' in result:
                for ref in result['uniProtKBCrossReferences']:
                    if ref.get('database') == 'GO':
                        go_id = ref.get('id', '')
                        go_term = ''
                        go_aspect = ''

                        for prop in ref.get('properties', []):
                            if prop.get('key') == 'GoTerm':
                                go_term = prop.get('value', '')

                        if go_id and go_term:
                            full_term = f"{go_id}: {go_term}"
                            go_terms.add(full_term)
                            go_count[full_term] += 1

                            if go_term.startswith('F:'):
                                go_by_category['Molecular Function'].append(full_term)
                            elif go_term.startswith('P:'):
                                go_by_category['Biological Process'].append(full_term)
                            elif go_term.startswith('C:'):
                                go_by_category['Cellular Component'].append(full_term)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(results)} entries...")

        return go_terms, go_by_category, go_count

    def analyze_go_terms(self, query: str = "reviewed:true AND length:[1 TO 60]") -> Dict:
        """Analyze GO terms and return statistics"""
        print("=" * 60)
        print("UniProt Short Peptide GO Terms Analysis")
        print("=" * 60)
        print()

        print("Fetching all results using stream endpoint...")
        print("-" * 60)

        results = self.stream_results(query)

        total_count = len(results)
        print(f"\nTotal proteins found: {total_count}")

        if total_count == 0:
            return {
                "total_proteins": 0,
                "total_go_terms": 0,
                "go_terms_by_category": {},
                "go_term_counts": {},
                "top_go_terms": {}
            }

        print("\nExtracting GO terms...")
        print("-" * 60)
        go_terms, go_by_category, go_count = self.extract_go_terms(results)

        print(f"\nUnique GO terms found: {len(go_terms)}")

        for category, terms in go_by_category.items():
            unique_terms = list(set(terms))
            print(f"  {category}: {len(unique_terms)} terms")

        top_go_terms = dict(go_count.most_common(20))

        return {
            "total_proteins": total_count,
            "total_go_terms": len(go_terms),
            "go_terms_by_category": {
                category: sorted(list(set(terms)))
                for category, terms in go_by_category.items()
            },
            "go_term_counts": dict(go_count),
            "top_go_terms": top_go_terms
        }

    def save_results(self, data: Dict, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {filename}")

    def print_summary(self, data: Dict):
        """Print a summary of the analysis"""
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total proteins analyzed: {data['total_proteins']}")
        print(f"Total unique GO terms: {data['total_go_terms']}")
        print()

        print("GO Terms by Category:")
        print("-" * 60)
        for category, terms in data['go_terms_by_category'].items():
            print(f"{category}: {len(terms)} terms")

        print("\nTop 20 Most Common GO Terms:")
        print("-" * 60)
        for i, (term, count) in enumerate(data['top_go_terms'].items(), 1):
            percentage = (count / data['total_proteins']) * 100
            print(f"{i}. {term} ({count} proteins, {percentage:.1f}%)")


def main():
    """Main function to query GO terms"""
    client = UniProtGOQuery(rate_limit=1.0)

    query = "reviewed:true AND length:[1 TO 60]"

    try:
        results = client.analyze_go_terms(query)
        client.print_summary(results)

        print("\n" + "=" * 60)

        output_file = "go_terms_analysis.json"
        client.save_results(results, output_file)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
