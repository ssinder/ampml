import requests
import time
import json
from typing import List, Dict, Set
from collections import Counter


class NonAntimicrobialPeptideSelector:
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

    def has_antimicrobial_go_terms(self, result: Dict) -> bool:
        """Check if protein has antimicrobial-related GO terms"""
        antimicrobial_go_terms = {
            'GO:0042742',  # defense response to bacterium
            'GO:0031640',  # killing of cells of another organism
            'GO:0006952',  # defense response
            'GO:0090729',  # toxin activity
            'GO:0045087',  # innate immune response
            'GO:0050832',  # defense response to fungus
            'GO:0050829',  # defense response to Gram-negative bacterium
            'GO:0050830',  # defense response to Gram-positive bacterium
        }

        if 'uniProtKBCrossReferences' in result:
            for ref in result['uniProtKBCrossReferences']:
                if ref.get('database') == 'GO':
                    go_id = ref.get('id', '')
                    if go_id in antimicrobial_go_terms:
                        return True

        return False

    def has_antimicrobial_keywords(self, result: Dict) -> bool:
        """Check if protein has antimicrobial-related keywords"""
        antimicrobial_keywords = {
            'Antimicrobial',
            'Defensin',
            'Antibiotic',
            'Bacteriocin',
            'Toxin',
            'Antiviral',
            'Allergen',
            'Antifungal',
            'Antiparasitic',
            'Immunity',
            'Innate immunity',
            'Defense',
            'Antibiotic resistance',
            'Antibiotic biosynthesis',
            'Plant defense',
            'Amphibian defense peptide',
            'Insect defense',
            'Fungicide',
            'Antimicrobial peptide',
        }

        if 'keywords' in result:
            for keyword_data in result['keywords']:
                keyword_name = keyword_data.get('name', '')
                for antimicrobial_keyword in antimicrobial_keywords:
                    if antimicrobial_keyword.lower() in keyword_name.lower():
                        return True

        return False

    def has_extracellular_location(self, result: Dict) -> bool:
        """Check if protein is located in extracellular region"""
        extracellular_go_terms = {
            'GO:0005576',  # extracellular region
            'GO:0005615',  # extracellular space
        }

        if 'uniProtKBCrossReferences' in result:
            for ref in result['uniProtKBCrossReferences']:
                if ref.get('database') == 'GO':
                    go_id = ref.get('id', '')
                    if go_id in extracellular_go_terms:
                        return True

        return False

    def calculate_antimicrobial_score(self, result: Dict) -> float:
        """Calculate a score indicating likelihood of being antimicrobial"""
        score = 0.0

        if self.has_antimicrobial_go_terms(result):
            score += 3.0

        if self.has_antimicrobial_keywords(result):
            score += 2.0

        if self.has_extracellular_location(result):
            score += 1.0

        return score

    def select_non_antimicrobial_peptides(self, query: str = "reviewed:true AND length:[1 TO 60]",
                                       top_n: int = 100) -> List[Dict]:
        """Select peptides least likely to be antimicrobial"""
        print("=" * 60)
        print("Non-Antimicrobial Peptide Selector")
        print("=" * 60)
        print()

        print("Fetching all results using stream endpoint...")
        print("-" * 60)

        results = self.stream_results(query)

        print(f"\nCalculating antimicrobial scores for {len(results)} proteins...")
        print("-" * 60)

        scored_peptides = []
        for i, result in enumerate(results):
            score = self.calculate_antimicrobial_score(result)

            peptide_info = {
                'accession': result.get('primaryAccession', ''),
                'protein_name': '',
                'organism': '',
                'length': 0,
                'sequence': '',
                'score': score,
                'go_terms': [],
                'keywords': []
            }

            if 'proteinDescription' in result:
                protein_desc = result['proteinDescription']
                if 'recommendedName' in protein_desc:
                    peptide_info['protein_name'] = protein_desc['recommendedName'].get('fullName', {}).get('value', '')

            if 'organism' in result:
                peptide_info['organism'] = result['organism'].get('scientificName', '')

            if 'sequence' in result:
                peptide_info['length'] = result['sequence'].get('length', 0)
                peptide_info['sequence'] = result['sequence'].get('value', '')

            if 'uniProtKBCrossReferences' in result:
                for ref in result['uniProtKBCrossReferences']:
                    if ref.get('database') == 'GO':
                        go_id = ref.get('id', '')
                        go_term = ''
                        for prop in ref.get('properties', []):
                            if prop.get('key') == 'GoTerm':
                                go_term = prop.get('value', '')
                        if go_id and go_term:
                            peptide_info['go_terms'].append(f"{go_id}: {go_term}")

            if 'keywords' in result:
                for keyword_data in result['keywords']:
                    peptide_info['keywords'].append(keyword_data.get('name', ''))

            scored_peptides.append(peptide_info)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(results)} entries...")

        scored_peptides.sort(key=lambda x: x['score'])

        print(f"\nSelected {top_n} peptides with lowest antimicrobial scores")
        print("-" * 60)

        selected_peptides = scored_peptides[:top_n]

        for i, peptide in enumerate(selected_peptides, 1):
            print(f"{i}. {peptide['accession']} - Score: {peptide['score']:.1f}")
            print(f"   Name: {peptide['protein_name']}")
            print(f"   Organism: {peptide['organism']}")
            print(f"   Length: {peptide['length']} aa")
            print(f"   Keywords: {', '.join(peptide['keywords'][:3])}")
            print()

        return selected_peptides

    def save_results(self, peptides: List[Dict], filename: str):
        """Save selected peptides to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(peptides, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {filename}")


def main():
    """Main function to select non-antimicrobial peptides"""
    selector = NonAntimicrobialPeptideSelector(rate_limit=1.0)

    query = "reviewed:true AND length:[1 TO 60]"

    try:
        selected_peptides = selector.select_non_antimicrobial_peptides(query, top_n=100)

        print("=" * 60)
        print("Summary")
        print("=" * 60)

        score_distribution = Counter(p['score'] for p in selected_peptides)
        print("Score distribution among selected peptides:")
        for score, count in sorted(score_distribution.items()):
            print(f"  Score {score:.1f}: {count} peptides")

        print("\n" + "=" * 60)

        output_file = "non_antimicrobial_peptides.json"
        selector.save_results(selected_peptides, output_file)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
