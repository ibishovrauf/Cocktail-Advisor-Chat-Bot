import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def process_cocktail_data():
    df = pd.read_csv('data/final_cocktails.csv', index_col=0)

    documents = []
    for _, row in df.iterrows():
        general_info = {
            'name': row['name'],
            'category': row['category'],
            'glassType': row['glassType'],
            'alcoholic': row['alcoholic'],
            'document_type': 'general_info',
            'cocktail_id': row.name,
            'content': f"Cocktail: {row['name']}\nCategory: {row['category']}\nGlass Type: {row['glassType']}\nAlcoholic: {row['alcoholic']}"
        }
        documents.append(general_info)
        

        ingredients_doc = {
            'name': row['name'],
            'ingredients': row['ingredients'],
            'ingredientMeasures': row['ingredientMeasures'],
            'document_type': 'ingredients',
            'cocktail_id': row.name,
            'content': f"Cocktail: {row['name']}\nIngredients: {row['ingredients']}\nIngredient Measures: {list(zip(eval(row['ingredients']), eval(row['ingredientMeasures'])))}"
        }
        documents.append(ingredients_doc)
        
        instructions_doc = {
            'name': row['name'],
            'instructions': row['instructions'],
            'document_type': 'instructions',
            'cocktail_id': row.name,
            'content': f"Cocktail: {row['name']}\nInstructions: {row['instructions']}"
        }
        documents.append(instructions_doc)
    return documents, df

def init_vector_db(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [doc['content'] for doc in documents]
    metadatas = [{k: v for k, v in doc.items() if k != 'content'} for doc in documents]
    
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    return vectorstore
