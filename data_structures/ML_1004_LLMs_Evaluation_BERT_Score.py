#!/usr/bin/env python3
"""
Tutorial Script: Evaluate a Llama3.2 Generated Candidate using a Custom BERTScore Implementation

This script demonstrates the following steps:
1. Loading a Llama3.2 model (for text generation) using the transformers library.
2. Generating candidate text from a prompt.
3. Loading a BERT model (e.g., "bert-base-uncased") from transformers to extract token embeddings.
4. Computing a simplified version of BERTScore by comparing candidate and reference embeddings via cosine similarity.

BERTScore Overview (Manual Implementation):
- For each token in the candidate, compute the cosine similarity with every token in the reference.
- For precision, average the maximum similarity for each candidate token.
- For recall, average the maximum similarity for each reference token.
- Compute the F1 score as the harmonic mean of precision and recall.
"""

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os 

# Set GPU environment and Hugging Face token
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'YOUR TOKEN'


def generate_candidate(prompt, model, tokenizer, max_length=100):
    """
    Generate candidate text using the Llama3.2 model.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate output (you can adjust generation parameters as needed)
    output_ids = model.generate(inputs["input_ids"], max_length=max_length, do_sample=True)
    candidate = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return candidate

def get_token_embeddings(text, tokenizer, model):
    """
    Given a text input, tokenize it and compute the model's last hidden states.
    Returns token embeddings and the list of tokens.
    """
    inputs = tokenizer(text, return_tensors="pt")
    # Get the last hidden states from the model.
    # `with torch.no_grad()` is used to disable gradient calculations.
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: [batch_size, sequence_length, hidden_dim]
    embeddings = outputs.last_hidden_state[0]  # take the first (and only) example
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return embeddings, tokens

def compute_cosine_similarity_matrix(candidate_emb, reference_emb):
    """
    Computes a cosine similarity matrix between candidate and reference embeddings.
    candidate_emb: Tensor of shape [n_candidate, hidden_dim]
    reference_emb: Tensor of shape [n_reference, hidden_dim]
    Returns: similarity matrix of shape [n_candidate, n_reference]
    """
    # Normalize the embeddings along the token dimension
    candidate_norm = candidate_emb / candidate_emb.norm(dim=1, keepdim=True)
    reference_norm = reference_emb / reference_emb.norm(dim=1, keepdim=True)
    # Compute cosine similarity matrix (dot product of normalized embeddings)
    sim_matrix = torch.matmul(candidate_norm, reference_norm.transpose(0, 1))
    return sim_matrix

def compute_bertscore(candidate, reference, bert_tokenizer, bert_model):
    """
    Computes a simplified BERTScore F1 between candidate and reference texts.
    This function computes token embeddings for both texts and then:
      - For precision: Averages the maximum cosine similarity for each candidate token.
      - For recall: Averages the maximum cosine similarity for each reference token.
      - F1 is the harmonic mean of precision and recall.
    """
    # Get token embeddings and tokens (you can ignore tokens that are special, if desired)
    candidate_emb, candidate_tokens = get_token_embeddings(candidate, bert_tokenizer, bert_model)
    reference_emb, reference_tokens = get_token_embeddings(reference, bert_tokenizer, bert_model)
    
    # Compute similarity matrix (each entry is cosine similarity between candidate & reference token)
    sim_matrix = compute_cosine_similarity_matrix(candidate_emb, reference_emb)
    
    # For each candidate token, find maximum similarity with any reference token (precision part)
    precision_scores, _ = torch.max(sim_matrix, dim=1)
    precision = precision_scores.mean().item()
    
    # For each reference token, find maximum similarity with any candidate token (recall part)
    recall_scores, _ = torch.max(sim_matrix, dim=0)
    recall = recall_scores.mean().item()
    
    # Compute F1 as harmonic mean of precision and recall
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

def main():
    # ---------------------------
    # 1. Load Llama3.2 Model for Generation
    # ---------------------------
    # Replace with the correct model identifier for Llama3.2
    llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"  
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id)

    # Define a prompt for the Llama3.2 model
    prompt = "Explain the theory of relativity in simple terms."
    candidate_text = generate_candidate(prompt, llama_model, llama_tokenizer)
    print("Candidate Generated by Llama3.2:")
    print(candidate_text)
    print("=" * 80)

    # ---------------------------
    # 2. Define the Ground Truth Reference
    # ---------------------------
    reference_text = ("The theory of relativity, developed by Albert Einstein, explains how time and space are linked for objects moving "
                      "at a consistent speed in a straight line. It introduced the idea that measurements of time and distance differ for observers "
                      "in different states of motion.")

    print("Reference Text:")
    print(reference_text)
    print("=" * 80)
    
    # ---------------------------
    # 3. Load a BERT Model for Evaluation (Embedding Extraction)
    # ---------------------------
    bert_model_name = "bert-base-uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)

    # ---------------------------
    # 4. Compute BERTScore (Precision, Recall, F1)
    # ---------------------------
    precision, recall, f1 = compute_bertscore(candidate_text, reference_text, bert_tokenizer, bert_model)
    print("BERTScore Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
