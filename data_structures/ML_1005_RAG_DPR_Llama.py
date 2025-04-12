import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import numpy as np
# Set GPU environment FIRST before any torch imports
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Critical for proper device mapping
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Now this will appear as cuda:0
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'YOUR TOKEN'

# Now import torch after setting CUDA variables
import torch

# Verify GPU assignment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing GPU: {torch.cuda.get_device_name(device)}\n")

# Load dataset
dataset = load_dataset("squad_v2")
train_subset = dataset['train'].select(range(500))
corpus = [item['context'] for item in train_subset] # This is your current "context database"


class DPRRetriever:
    def __init__(self, 
                 q_encoder_name="facebook/dpr-question_encoder-single-nq-base",
                 ctx_encoder_name="facebook/dpr-ctx_encoder-single-nq-base"):
        
        # Initialize models on target GPU
        self.question_encoder = DPRQuestionEncoder.from_pretrained(q_encoder_name).to(device)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(q_encoder_name)
        self.corpus_embeddings = None
        self.corpus = []

    def _encode_questions(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)  # Move inputs to GPU
        
        with torch.no_grad():
            return self.question_encoder(**inputs).pooler_output

    def _encode_contexts(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)  # Move inputs to GPU
        
        with torch.no_grad():
            return self.ctx_encoder(**inputs).pooler_output

    def fit(self, corpus):
        """Encode and store all context passages"""
        self.corpus = corpus
        batch_size = 16
        
        all_embeddings = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            embeddings = self._encode_contexts(batch)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())  # Store in CPU memory
        
        self.corpus_embeddings = torch.tensor(np.concatenate(all_embeddings)).to(device)

    def retrieve(self, query, top_k=3):
        """Retrieve documents using DPR"""
        # Encode query
        query_embedding = self._encode_questions([query])
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

        # Compute similarities on GPU
        scores = torch.mm(query_embedding, self.corpus_embeddings.T)
        top_indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()[0]

        return [(self.corpus[idx], scores[0][idx].item()) for idx in top_indices]

class LlamaRAGChatbot:
    def __init__(self, retriever, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model directly to target device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device}  # Force to specific device
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def format_prompt(self, query, context):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant that answers questions based on provided context.
        Use only the information from the context to answer. If unsure, say "I don't know".<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Context: {context}
        Question: {query}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        Answer: """
    
    def generate_response(self, query, top_k=3):
        try:
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(query, top_k)
            context = " ".join([doc[0] for doc in retrieved_docs])
            
            # Format prompt
            prompt = self.format_prompt(query, context)
            
            # Tokenize and generate on correct device
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response.split("Answer:")[-1].strip()
            
            return answer
            
        except Exception as e:
            print(f"Generation Error: {e}")
            return "I'm having trouble answering that right now."

if __name__ == "__main__":
    # Initialize and run
    retriever = DPRRetriever()
    retriever.fit([ex['context'] for ex in dataset['train'].select(range(100))])
    
    rag_chatbot = LlamaRAGChatbot(retriever)
    
    response = rag_chatbot.generate_response("Who was the first president of the United States?")
    print(f"\nFinal Answer: {response}")

# can replace it to faiss

