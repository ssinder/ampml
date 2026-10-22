import requests
import time
import csv
import json
from typing import List, Dict


class ShortPeptideCSVExporter:
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

    def extract_peptide_data(self, result: Dict) -> Dict:
        """Extract all relevant data from a single peptide entry"""
        peptide_data = {
            'accession': '',
            'entry_name': '',
            'protein_name': '',
            'gene_names': '',
            'organism': '',
            'organism_id': '',
            'length': 0,
            'sequence': '',
            'mass': 0,
            'keywords': '',
            'go_terms': '',
            'go_molecular_function': '',
            'go_biological_process': '',
            'go_cellular_component': '',
            'antimicrobial_go_terms': '',
            'antimicrobial_keywords': '',
            'extracellular_location': '',
            'reviewed': '',
            'protein_existence': ''
        }

        peptide_data['accession'] = result.get('primaryAccession', '')

        if 'uniProtKbId' in result:
            peptide_data['entry_name'] = result['uniProtKbId']

        if 'proteinDescription' in result:
            protein_desc = result['proteinDescription']
            if 'recommendedName' in protein_desc:
                peptide_data['protein_name'] = protein_desc['recommendedName'].get('fullName', {}).get('value', '')

        if 'genes' in result:
            gene_names = []
            for gene in result['genes']:
                if 'geneName' in gene:
                    gene_names.append(gene['geneName'].get('value', ''))
            peptide_data['gene_names'] = '; '.join(gene_names)

        if 'organism' in result:
            peptide_data['organism'] = result['organism'].get('scientificName', '')
            peptide_data['organism_id'] = result['organism'].get('taxonId', '')

        if 'sequence' in result:
            peptide_data['length'] = result['sequence'].get('length', 0)
            peptide_data['sequence'] = result['sequence'].get('value', '')
            peptide_data['mass'] = result['sequence'].get('molWeight', 0)

        if 'keywords' in result:
            keywords = []
            for keyword_data in result['keywords']:
                keywords.append(keyword_data.get('name', ''))
            peptide_data['keywords'] = '; '.join(keywords)

        if 'uniProtKBCrossReferences' in result:
            go_terms = []
            go_mf = []
            go_bp = []
            go_cc = []
            antimicrobial_go_terms = []

            for ref in result['uniProtKBCrossReferences']:
                if ref.get('database') == 'GO':
                    go_id = ref.get('id', '')
                    go_term = ''
                    go_aspect = ''

                    for prop in ref.get('properties', []):
                        if prop.get('key') == 'GoTerm':
                            go_term = prop.get('value', '')
                            if go_term.startswith('F:'):
                                go_aspect = 'F'
                            elif go_term.startswith('P:'):
                                go_aspect = 'P'
                            elif go_term.startswith('C:'):
                                go_aspect = 'C'

                    if go_id and go_term:
                        full_term = f"{go_id}: {go_term}"
                        go_terms.append(full_term)

                        if go_aspect == 'F':
                            go_mf.append(full_term)
                        elif go_aspect == 'P':
                            go_bp.append(full_term)
                        elif go_aspect == 'C':
                            go_cc.append(full_term)

                        antimicrobial_go_ids = {
                            'GO:0042742',  # defense response to bacterium
                            'GO:0031640',  # killing of cells of another organism
                            'GO:0006952',  # defense response
                            'GO:0090729',  # toxin activity
                            'GO:0045087',  # innate immune response
                            'GO:0050832',  # defense response to fungus
                            'GO:0050829',  # defense response to Gram-negative bacterium
                            'GO:0050830',  # defense response to Gram-positive bacterium
                        }

                        if go_id in antimicrobial_go_ids:
                            antimicrobial_go_terms.append(full_term)

            peptide_data['go_terms'] = '; '.join(go_terms)
            peptide_data['go_molecular_function'] = '; '.join(go_mf)
            peptide_data['go_biological_process'] = '; '.join(go_bp)
            peptide_data['go_cellular_component'] = '; '.join(go_cc)
            peptide_data['antimicrobial_go_terms'] = '; '.join(antimicrobial_go_terms)

        if 'keywords' in result:
            antimicrobial_keywords = []
            antimicrobial_keyword_list = {
                'Antimicrobial', 'Defensin', 'Antibiotic', 'Bacteriocin',
                'Toxin', 'Antiviral', 'Allergen', 'Antifungal',
                'Antiparasitic', 'Immunity', 'Innate immunity', 'Defense',
                'Antibiotic resistance', 'Antibiotic biosynthesis', 'Plant defense',
                'Amphibian defense peptide', 'Insect defense', 'Fungicide',
                'Antimicrobial peptide'
            }

            for keyword_data in result['keywords']:
                keyword_name = keyword_data.get('name', '')
                for antimicrobial_keyword in antimicrobial_keyword_list:
                    if antimicrobial_keyword.lower() in keyword_name.lower():
                        antimicrobial_keywords.append(keyword_name)
                        break

            peptide_data['antimicrobial_keywords'] = '; '.join(antimicrobial_keywords)

        if 'uniProtKBCrossReferences' in result:
            extracellular_go_terms = {
                'GO:0005576',  # extracellular region
                'GO:0005615',  # extracellular space
            }

            for ref in result['uniProtKBCrossReferences']:
                if ref.get('database') == 'GO':
                    go_id = ref.get('id', '')
                    if go_id in extracellular_go_terms:
                        peptide_data['extracellular_location'] = 'Yes'
                        break

            if not peptide_data['extracellular_location']:
                peptide_data['extracellular_location'] = 'No'

        peptide_data['reviewed'] = 'Yes' if result.get('entryType') == 'REVIEWED' else 'No'

        if 'proteinExistence' in result:
            protein_existence = result['proteinExistence']
            if isinstance(protein_existence, dict):
                peptide_data['protein_existence'] = protein_existence.get('value', '')
            else:
                peptide_data['protein_existence'] = str(protein_existence)

        return peptide_data

    def export_to_csv(self, query: str = "reviewed:true AND length:[1 TO 60]",
                   output_file: str = "short_peptides_complete.csv"):
        """Export all short peptides to CSV file"""
        print("=" * 60)
        print("Short Peptide CSV Exporter")
        print("=" * 60)
        print()

        print("Fetching all results using stream endpoint...")
        print("-" * 60)

        results = self.stream_results(query)

        print(f"\nExtracting data for {len(results)} proteins...")
        print("-" * 60)

        all_peptides = []
        for i, result in enumerate(results):
            peptide_data = self.extract_peptide_data(result)
            all_peptides.append(peptide_data)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(results)} entries...")

        print(f"\nWriting data to CSV file: {output_file}")
        print("-" * 60)

        fieldnames = [
            'accession', 'entry_name', 'protein_name', 'gene_names', 'organism', 'organism_id',
            'length', 'sequence', 'mass', 'keywords', 'go_terms',
            'go_molecular_function', 'go_biological_process', 'go_cellular_component',
            'antimicrobial_go_terms', 'antimicrobial_keywords', 'extracellular_location',
            'reviewed', 'protein_existence'
        ]

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_peptides)

        print(f"Successfully exported {len(all_peptides)} peptides to {output_file}")

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        antimicrobial_go_count = sum(1 for p in all_peptides if p['antimicrobial_go_terms'])
        antimicrobial_keyword_count = sum(1 for p in all_peptides if p['antimicrobial_keywords'])
        extracellular_count = sum(1 for p in all_peptides if p['extracellular_location'] == 'Yes')

        print(f"Total peptides: {len(all_peptides)}")
        print(f"Peptides with antimicrobial GO terms: {antimicrobial_go_count} ({antimicrobial_go_count/len(all_peptides)*100:.1f}%)")
        print(f"Peptides with antimicrobial keywords: {antimicrobial_keyword_count} ({antimicrobial_keyword_count/len(all_peptides)*100:.1f}%)")
        print(f"Peptides with extracellular location: {extracellular_count} ({extracellular_count/len(all_peptides)*100:.1f}%)")

        return all_peptides


def main():
    """Main function to export short peptides to CSV"""
    exporter = ShortPeptideCSVExporter(rate_limit=1.0)

    query = "reviewed:true AND length:[1 TO 60]"

    try:
        all_peptides = exporter.export_to_csv(query, "short_peptides_complete.csv")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
