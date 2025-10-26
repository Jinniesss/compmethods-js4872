# CBB 6340 - Problem Set 3

### Student Information

Name: Jinnie Sun

NetID: js4872

### Instructions for Running Scripts

This project's scripts are written in Python.

### Exercise Answers & Results

#### Exercise 1: Retrieving PubMed Data via Entrez API (25 points)

##### 1a. Retrieve PubMed IDs for Alzheimer’s and Cancer Papers

```python
def get_pubmed_id_lists(term):
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"esearch.fcgi?db=pubmed&term={term}&retmode=xml&retmax=1000"
    )
    doc = m.parseString(r.text)
    id_lists = doc.getElementsByTagName("Id")
    id_list = [id_node.childNodes[0].wholeText for id_node in id_lists]
    return id_list
  
if __name__ == "__main__":
    # 1a. Retrieve PubMed id_lists for Alzheimer’s and Cancer Papers
    terms = ['Alzheimers+AND+2024[pdat]', 'cancer+AND+2024[pdat]']
    id_lists = {}
    for term in terms:
        id_lists[term.split('+')[0]] = get_pubmed_id_lists(term)
```

##### 1b. Retrieve Metadata for the Papers

```python
def get_pubmed_metadata(query, id_list):
    output_filename = f"problem_set_3/metadata_{query}.json"
    # if os.path.exists(output_filename):
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

        abstract_element = article.find(".//AbstractText")
        abstract_text = ""
        if abstract_element is not None:
            abstract_text = ET.tostring(abstract_element, method="text", encoding="unicode").strip()

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
		# 1b. Retrieve Metadata for the Papers
    for query, id_list in id_lists.items():
        get_pubmed_metadata(query, id_list)
```

I noticed that though 1000 ids were retrieved by esearch for each query, only 999 articles were found by the efetch in the case of 'cancer'. This one pubmed id might be retracted or suppressed for some reason.

Head of `metadata_Alzheimers.json`:

```json
{
    "41058862": {
        "ArticleTitle": "Deep learning assessment of disproportionately enlarged subarachnoid-space hydrocephalus in Hakim's disease or idiopathic normal pressure hydrocephalus.",
        "AbstractText": "Disproportionately enlarged subarachnoid-space hydrocephalus (DESH) is a key feature of Hakim's disease (synonymous with idiopathic normal pressure hydrocephalus; iNPH). However, it previously had been only subjectively evaluated.",
        "query": "Alzheimers"
    },
    "41024939": {
    ...
```

Head of `metadata_cancer.json`:

```json
{
    "41134009": {
        "ArticleTitle": "Light in the darkness - accessibility to palliative care for cancer patients of Chinese background and their families.",
        "AbstractText": "Palliative care addresses the comprehensive needs of advanced-stage cancer patients, enhancing their quality of life. However, due to cultural and linguistic differences, Chinese Australians encounter substantial barriers to accessing these services.",
        "query": "cancer"
    },
    "41111483": {
      ...
```

##### 1c. Analyze Overlap Between the Two Paper Sets

```python
# 1c. Analyze Overlap Between the Two Paper Sets
overlap = set(id_lists['Alzheimers']).intersection(set(id_lists['cancer']))
print(f"Overlap between Alzheimer's and Cancer papers: {len(overlap)}")

# Print the titles of the overlapped papers
for pmid in overlap:
    with open("problem_set_3/metadata_Alzheimers.json", "r") as f:
        alz_metadata = json.load(f)
    title = alz_metadata[pmid]["ArticleTitle"]
    print(f"PMID: {pmid}, Title: {title}")
```

Output:

```
Overlap between Alzheimer's and Cancer papers: 4
PMID: 40949928, Title: MedGraphNet: Leveraging Multi-Relational Graph Neural Networks and Text Knowledge for Biomedical Predictions.
PMID: 40326981, Title: Exosomes in neurodegenerative diseases: Therapeutic potential and modification methods.
PMID: 40800467, Title: Tractography from T1-weighted MRI: Empirically exploring the clinical viability of streamline propagation without diffusion MRI.
PMID: 40395755, Title: Tau-targeting nanoparticles for treatment of Alzheimer's disease.
```

##### 1d. Handle Structured Abstracts

All `<AbstractText>` parts within the parent `<Abstract>` tag are found. Then labels like 'METHODS' are prepended. Finally all sections are concatenated with spaces.

Tested with the example PMID:20966393. ( [metadata_test.json](metadata_test.json) )

```python
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
```

Limitation: It's assumed that the structure is defined by the `Label` attribute. If an abstract is formatted without this attribute, this code would not capture the section titles. 



