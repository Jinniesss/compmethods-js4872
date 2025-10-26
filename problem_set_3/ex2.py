import tqdm
from transformers import AutoTokenizer, AutoModel
import json

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

def get_abstract(paper):
    return paper['AbstractText']

json_files = ['problem_set_3/metadata_Alzheimers.json', 'problem_set_3/metadata_cancer.json']
papers = {}
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        cur_papers = json.load(f)
        papers.update(cur_papers)

embeddings = {}
for pmid, paper in tqdm.tqdm(papers.items()):
    data = [paper["ArticleTitle"] + tokenizer.sep_token + get_abstract(paper)]
    inputs = tokenizer(
        data, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

# turn our dictionary into a list
embeddings = [embeddings[pmid] for pmid in papers.keys()]

print('end')