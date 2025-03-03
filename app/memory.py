import re
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document

class MemoryManager:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever()
        self.memory = VectorStoreRetrieverMemory(retriever=self.retriever)
        
    def detect_preferences(self, user_message):
        favorite_ingredients = re.findall(r"my favorite ingredients? (?:is|are) (.*)", user_message, re.IGNORECASE)
        favorite_cocktails = re.findall(r"my favorite cocktails? (?:is|are) (.*)", user_message, re.IGNORECASE)
        
        preferences = []
        
        if favorite_ingredients:
            for ingredients in favorite_ingredients:
                preferences.append(Document(
                    page_content=f"User's favorite ingredients: {ingredients}",
                    metadata={"type": "favorite_ingredients"}
                ))
                
        if favorite_cocktails:
            for cocktails in favorite_cocktails:
                preferences.append(Document(
                    page_content=f"User's favorite cocktails: {cocktails}",
                    metadata={"type": "favorite_cocktails"}
                ))
        
        if preferences:
            self.vectorstore.add_documents(preferences)
            
        return bool(preferences)
    
    def get_preferences(self, query_type="all", k=5):
        if query_type == "all":
            return self.retriever.get_relevant_documents("User's favorite")
        else:
            return self.retriever.get_relevant_documents(f"User's favorite {query_type}")