import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os

# Set GPU environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'YOUR TOKEN'

# Load dataset (using SQuAD v2 for better question variety)
dataset = load_dataset("squad_v2")  # Updated to SQuAD v2

# Use a meaningful subset with questions and contexts
train_subset = dataset['train'].select(range(500))
corpus = [item['context'] for item in train_subset]
questions = [item['question'] for item in train_subset]  # For better query examples

# 1. Enhanced Retriever with better similarity handling
class Retriever:
    def __init__(self, model_name='all-mpnet-base-v2'):  # Stronger default model
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = []
    
    def fit(self, corpus):
        self.corpus = corpus
        # Normalize embeddings for better cosine similarity calculation
        embeddings = self.model.encode(corpus, convert_to_tensor=True)
        self.corpus_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        
        # Use PyTorch for GPU acceleration
        similarities = torch.mm(query_embedding, self.corpus_embeddings.T)
        top_k_indices = torch.topk(similarities, k=top_k, dim=1).indices.cpu().numpy()[0]
        
        retrieved_docs = [(self.corpus[idx], similarities[0][idx].item()) for idx in top_k_indices]
        
        print("\nRetrieved Contexts:")
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            print(f"Context {i} (score: {score:.4f}): {doc[:100]}...")
            print("=====")
        
        return retrieved_docs

# 2. Improved Generator with better prompting
class RAGChatbot:
    def __init__(self, retriever, generator_model_name='google/flan-t5-base'):  # Better base model
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self.generator.to(self.device)
    
    def generate_response(self, query, top_k=3, max_length=512):
        try:
            retrieved_docs = self.retriever.retrieve(query, top_k)
            if not retrieved_docs:
                return "No relevant information found."
            
            # Smart context truncation
            context = self._truncate_context([doc[0] for doc in retrieved_docs], max_length)
            
            # Better prompt engineering
            input_text = f"answer based on context: {context} question: {query}"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Improved generation parameters
            outputs = self.generator.generate(
                inputs.input_ids,
                max_new_tokens=150,
                num_beams=1,  # Better beam search
                early_stopping=True,
                repetition_penalty=2.5,
                length_penalty=1.2
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nFinal Answer: {answer}")
            return answer
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."

    def _truncate_context(self, contexts, max_length):
        """Ensure combined context fits within model's max length"""
        combined = ""
        for context in contexts:
            if len(self.tokenizer.tokenize(combined + context)) < max_length:
                combined += context + " "
            else:
                remaining = max_length - len(self.tokenizer.tokenize(combined))
                if remaining > 100:  # Only add if significant space remains
                    combined += self.tokenizer.decode(
                        self.tokenizer.encode(context, max_length=remaining, truncation=True)
                    ) + " "
                break
        return combined.strip()

# Example usage with error handling meta-llama/Llama-3.2-1B-Instruct google/flan-t5-base
if __name__ == "__main__":
    retriever = Retriever()
    retriever.fit(corpus)
    
    rag_chatbot = RAGChatbot(retriever)
    
    # Use a question that exists in the dataset subset
    input_query = "Who was the first president of the United States?"
    
    response = rag_chatbot.generate_response(input_query)
    print(f"\nChatbot Response: {response}")