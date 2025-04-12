#!/usr/bin/env python
import re
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset

# For embedding-based retrieval
from sentence_transformers import SentenceTransformer
import faiss


# Set GPU environment and Hugging Face token
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'YOUR TOKEN'

class RAGEvaluator:
    def __init__(self):
        # Initialize generator model (for QA generation)
        self.generator_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Initialize judge model with 4-bit quantization
        self.judge_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        )
        
        # Initialize reader model (for simulating a RAG system)
        self.reader_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.reader_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.reader_model.to("cuda")
        self.reader_model.eval()
        
        # Build a context corpus from a knowledge base.
        # Here we use a subset of "m-ric/huggingface_doc" as our corpus.
        use_customized_dataset = False

        if use_customized_dataset:
            # Custom dataset with your curated Q&A content
            custom_data = [
                {"text": "The Hugging Face library provides tools for deploying machine learning models. For example, the `Inference Endpoints` service allows users to host models as APIs. This is distinct from the Transformers library, which focuses on model architecture and training."},
                {"text": "Key differences: Hugging Face is a platform for model sharing and deployment, while Transformers is a Python library for building NLP models like BERT and GPT."}
            ]
            self.corpus = [item["text"] for item in custom_data]
        else:
            # Default dataset from Hugging Face
            dataset = load_dataset("m-ric/huggingface_doc", split="train")
            train_subset = dataset.select(range(500))
            self.corpus = [item["text"] for item in train_subset]

        # Build embedding retrievr
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_embeddings = self.embedder.encode(
            self.corpus, 
            convert_to_tensor=False,
            show_progress_bar=True
        )

        # Create FAISS index
        self.dim = corpus_embeddings.shape[1]  # More robust dimension extraction
        corpus_embeddings_np = np.array(corpus_embeddings).astype("float32")

        # Use IndexFlatIP for cosine similarity instead of L2 distance
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(corpus_embeddings_np)  # Normalize for cosine similarity
        self.index.add(corpus_embeddings_np)

        print(f"FAISS index built with {self.index.ntotal} documents.")

    def generate_qa_pairs(self, num_pairs=4, max_retries=3):
        """Generate QA pairs with retry logic and filter out prompt examples"""
        # Define placeholder and example texts to filter.
        prequQ = "<Your question ending with a question mark>"
        prequA = "<Your answer in 1-2 sentences>"
        example_question = "What is the Transformers library?"
        example_answer = "It is a Python library for state-of-the-art machine learning architectures."
        
        forbidden_questions = {prequQ.lower(), example_question.lower()}
        forbidden_answers = {prequA.lower(), example_answer.lower()}
        
        all_valid_pairs = []
        for attempt in range(max_retries):
            try:
                prompt = f"""Generate exactly {num_pairs} question-answer pairs about Hugging Face libraries.
Each pair must be in the following format (one pair per line) with no extra text:
Q: {prequQ}
A: {prequA}

Example:
Q: {example_question}
A: {example_answer}
"""
                inputs = self.generator_tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.3,
                    do_sample=True,
                    num_return_sequences=3,
                    pad_token_id=self.generator_tokenizer.eos_token_id
                )
                
                for i in range(outputs.shape[0]):
                    generated_text = self.generator_tokenizer.decode(outputs[i], skip_special_tokens=True)
                    print("Generated QA text (sequence {}):\n{}".format(i+1, generated_text))
                    qa_pattern = r"(?i)(?:Q:)\s*(.*?)\s*(?:\n|$)(?:A:)\s*(.*?)(?=\nQ:|\Z)"
                    matches = re.findall(qa_pattern, generated_text, re.DOTALL)
                    for q, a in matches:
                        q_clean = q.strip()
                        a_clean = a.strip()
                        if q_clean.lower() in forbidden_questions or a_clean.lower() in forbidden_answers:
                            continue
                        if not q_clean.endswith("?"):
                            q_clean += "?"
                        if 8 < len(q_clean) < 150 and 15 < len(a_clean) < 300:
                            pair = {"question": q_clean, "answer": a_clean}
                            if pair not in all_valid_pairs:
                                all_valid_pairs.append(pair)
                        if len(all_valid_pairs) >= num_pairs:
                            break
                    if len(all_valid_pairs) >= num_pairs:
                        break
            except Exception as e:
                print(f"Generation attempt {attempt+1} failed: {str(e)}")
            if len(all_valid_pairs) >= num_pairs:
                break
        return all_valid_pairs[:num_pairs]

    def retrieve_context(self, question, top_k=1):
        """
        Use an embedding-based retriever to find the most relevant context from the corpus.
        """
        q_embedding = self.embedder.encode([question], convert_to_tensor=False) # is_causal is False , attention_mask is None
        q_embedding_np = np.array(q_embedding).astype("float32")
        distances, indices = self.index.search(q_embedding_np, top_k)
        # Return the top context. For more complex scenarios, you can return top_k contexts.
        best_index = indices[0][0]
        return self.corpus[best_index]

    def generate_rag_response(self, question):
        """
        Generate a response using the reader model.
        The reader receives a retrieved context (from our FAISS index) along with the question.
        """
        context = self.retrieve_context(question)
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        inputs = self.reader_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.reader_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        answer = self.reader_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer, context

    def evaluate_response(self, question, reference, response):
        """Evaluate a test response relative to the reference answer using the judge model"""
        try:
            prompt = f"""You are an expert evaluator. Do not repeat the instructions.
    Evaluate the test response compared to the reference answer for the given question.
    Provide a numerical score between 1 and 5 (1 = completely incorrect, 5 = perfect match) and a brief explanation.
    Output exactly in the following format (with no extra text):
    Score: [a single digit between 1 and 5]
    Reasoning: <your explanation>

    Now evaluate:
    Question: {question}
    Reference Answer: {reference}
    Test Response: {response}
    """
            inputs = self.judge_tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=False
            )
            result = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Raw judge output:\n", result)
            
            # Use regex to extract score and reasoning from the result.
            # This pattern looks for "Score:" followed by a number and then "Reasoning:" followed by everything.
            pattern = r"Score:\s*(\d+)\s*Reasoning:\s*(.*)"
            match = re.search(pattern, result, re.DOTALL)
            if match:
                score = int(match.group(1))
                reasoning = match.group(2).strip()
            else:
                score = -1
                reasoning = "Evaluation unavailable"
            
            return {
                "question": question,
                "reference": reference,
                "response": response,
                "score": score,
                "reasoning": reasoning[:200]  # Truncate if too long
            }
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return {
                "question": question,
                "reference": reference,
                "response": response,
                "score": -1,
                "reasoning": "Evaluation error"
            }


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # Step 1: Generate QA pairs using the generator model.
    print("Generating QA pairs...")
    qa_pairs = evaluator.generate_qa_pairs(num_pairs=4, max_retries=5)
    if not qa_pairs:
        print("Error: Failed to generate QA pairs after multiple attempts")
        exit(1)
    
    print("QA Pairs:")
    for pair in qa_pairs:
        print(f"Q: {pair['question']}\nA: {pair['answer']}\n")
    
    # Step 2: Generate RAG responses using the reader model with dynamic retrieval.
    print("Generating RAG responses for each question...")
    rag_responses = []
    for pair in qa_pairs:
        response, retrieved_context = evaluator.generate_rag_response(pair["question"])
        rag_responses.append(response)
        print(f"\nQuestion: {pair['question']}")
        print(f"Retrieved Context: {retrieved_context[:200]}...")  # Print first 200 chars
        print(f"RAG Response: {response}")
        # retrieved_context = evaluator.retrieve_context(pair["question"])

    
    # Step 3: Evaluate each RAG response against the reference answer.
    print("Evaluating RAG responses...")
    results = []
    for index, (pair, response) in enumerate(zip(qa_pairs, rag_responses)):
        result = evaluator.evaluate_response(pair["question"], pair["answer"], response)
        results.append(result)
        print(f"\nQA ID: {index}")
        print(f"\nQuestion: {result['question']}")
        print(f"\nReference: {result['reference']}")
        print(f"\nResponse: {result['response']}")
        print(f"\nScore: {result['score']}/5")
        print(f"\nReasoning: {result['reasoning']}")
        print("="*15)
        print()
    
    if results:
        df = pd.DataFrame(results)
        print("\nEvaluation Report:")
        print(df[["question", "reference", "response", "score", "reasoning"]].to_markdown(index=False))
        print(f"\nAverage Score: {df['score'].mean():.1f}/5")
    else:
        print("Error: No evaluation results generated")