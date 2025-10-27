import requests
import json
import os
import sys
import xml.dom.minidom as m
import xml.etree.ElementTree as ET

def get_pubmed_id_lists(term):
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"esearch.fcgi?db=pubmed&term={term}&retmode=xml&retmax=1000"
    )
    doc = m.parseString(r.text)
    id_lists = doc.getElementsByTagName("Id")
    id_list = [id_node.childNodes[0].wholeText for id_node in id_lists]
    return id_list

def get_pubmed_metadata(query, id_list):
    output_filename = f"problem_set_3/metadata_{query}.json"
    # if os.path.exists(output_filename) and query != 'test':
    #     return
    id_list_str = ",".join(id_list)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    parameters = {
        "db": "pubmed",
        "id": id_list_str,
        "retmode": "xml"
    }
    r = requests.post(url, data=parameters)

    root = ET.fromstring(r.text)
    metadata_dict = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_element = article.find(".//PMID")
        pmid = pmid_element.text

        title_element = article.find(".//ArticleTitle")
        title_text = ""
        if title_element is not None:
            title_text = ET.tostring(title_element, method="text", encoding="unicode").strip()

        # abstract_element = article.find(".//AbstractText")
        # abstract_text = ""
        # if abstract_element is not None:
        #     abstract_text = ET.tostring(abstract_element, method="text", encoding="unicode").strip()
            
        # 1d. Handle Structured Abstracts
        abstract_parent = article.find(".//Abstract")
        abstract_text = ""
        
        if abstract_parent is not None:
            parts = []
            for part in abstract_parent.findall(".//AbstractText"):
                part_text = ET.tostring(part, method="text", encoding="unicode").strip()
                if 'Label' in part.attrib:
                    parts.append(f"{part.attrib['Label']}: {part_text}")
                else:
                    parts.append(part_text)
            
            abstract_text = " ".join(parts)
            
        metadata_dict[pmid] = {
            "ArticleTitle": title_text,
            "AbstractText": abstract_text,
            "query": query 
        }

    with open(output_filename, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"Metadata for {query} saved to {output_filename}") 
    return metadata_dict

if __name__ == "__main__":
    # 1a. Retrieve PubMed id_lists for Alzheimer’s and Cancer Papers
    terms = ['Alzheimers+AND+2024[pdat]', 'cancer+AND+2024[pdat]']
    id_lists = {}
    for term in terms:
        id_lists[term.split('+')[0]] = get_pubmed_id_lists(term)
    
    id_lists['test'] = ['20966393']
    # 1b. Retrieve Metadata for the Papers
    for query, id_list in id_lists.items():
        get_pubmed_metadata(query, id_list)

    # 1c. Analyze Overlap Between the Two Paper Sets
    overlap = set(id_lists['Alzheimers']).intersection(set(id_lists['cancer']))
    print(f"Overlap between Alzheimer's and Cancer papers: {len(overlap)}")
    
    # Print the titles of the overlapped papers
    for pmid in overlap:
        with open("problem_set_3/metadata_Alzheimers.json", "r") as f:
            alz_metadata = json.load(f)
        title = alz_metadata[pmid]["ArticleTitle"]
        print(f"PMID: {pmid}, Title: {title}")