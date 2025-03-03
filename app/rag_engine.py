from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class RAGEngine:
    def __init__(self, vectorstore, memory_manager):
        self.vectorstore = vectorstore
        self.memory_manager = memory_manager
        self.llm = OpenAI()

    def query(self, user_input):
        self.memory_manager.detect_preferences(user_input)

        doc_type_priority = None
        if any(term in user_input.lower()
               for term in ['ingredient', 'contain', 'made with']):
            doc_type_priority = 'ingredients'
        elif any(term in user_input.lower()
                 for term in ['how to make', 'instruction', 'preparation']):
            doc_type_priority = 'instructions'

        if doc_type_priority:
            relevant_docs = self.vectorstore.similarity_search(
                user_input, k=15, filter={"document_type": doc_type_priority})
        else:
            relevant_docs = self.vectorstore.similarity_search(user_input, k=8)

        cocktails_info = {}
        for doc in relevant_docs:
            cocktail_id = doc.metadata.get('cocktail_id')
            if cocktail_id not in cocktails_info:
                cocktails_info[cocktail_id] = []
            cocktails_info[cocktail_id].append(doc)

        top_cocktails = list(cocktails_info.values())[:5]

        user_preferences = self.memory_manager.get_preferences()

        context_parts = []
        for cocktail_docs in top_cocktails:
            cocktail_context = "\n".join(
                [doc.page_content for doc in cocktail_docs])
            context_parts.append(cocktail_context)

        context = "\n\n".join(context_parts)

        preferences_context = "\n\n".join(
            [doc.page_content for doc in user_preferences])

        prompt_template = """
        You are a cocktail expert assistant. Answer the user's question based on the following information:
        
        COCKTAIL DATABASE:
        {context}
        
        USER PREFERENCES:
        {preferences}
        
        Question: {question}
        
        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "preferences", "question"])

        chain = LLMChain(llm=self.llm, prompt=prompt)

        response = chain.run(context=context,
                             preferences=preferences_context,
                             question=user_input)
        return response
