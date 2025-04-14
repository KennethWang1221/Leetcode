#!/usr/bin/env python
import re
import os
import torch
import pandas as pd
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset
from collections import Counter
from fractions import Fraction
# For embedding-based retrieval
from sentence_transformers import SentenceTransformer
import faiss
import torch.nn.functional as F


# Set GPU environment and Hugging Face token
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'Your Token'

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

        self.pp_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
        self.pp_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-3.2-1B-Instruct',
            torch_dtype=torch.bfloat16,
        )
        self.pp_model.to("cuda")
        self.pp_model.eval()

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


    def calculate_perplexity(self, text, manually=True):
        """
        Compute the perplexity for a given text using the model.
        
        There are two modes controlled by the 'manually' flag:
        - If manually=True: the function computes the perplexity manually by:
            1. Encoding the text and obtaining logits.
            2. Shifting the logits and true labels so that we predict token t from tokens < t.
            3. Manually computing softmax probabilities, token-level cross-entropy loss,
                averaging them, and computing perplexity as exp(average loss).
        - If manually=False: the function computes perplexity using the model's built-in loss,
            which automatically handles shifting.
        
        Parameters:
            text (str): The input text.
            manually (bool): If True, use manual calculation; if False, use the model's loss.
        
        Returns:
            float: The computed perplexity.
        """
        # Encode the text.
        input_ids = self.pp_tokenizer.encode(text, return_tensors="pt").to("cuda")
        
        if manually:
            ignore_index = -100
            # Step 2: Forward  pass to obtain logits.
            with torch.no_grad():
                outputs = self.pp_model(input_ids)
                logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
            logits = logits.float()
            vocab_size = logits.size(-1)
            
            # Step 3: Pad the input IDs with the ignore index so that we can shift them.
            # Padding on the right will result in a tensor of shape [batch, seq_len+1].
            padded_labels = F.pad(input_ids, (0, 1), value=ignore_index)
            
            # Shift the labels: Remove the first token so that targets become tokens 1...L.
            shifted_true_labels = padded_labels[:, 1:].contiguous()  # Shape: [batch, seq_len]
            
            # Step 4: Align logits with the shifted labels.
            shifted_logits = logits[:, :shifted_true_labels.size(1), :]  # Shape: [batch, seq_len, vocab_size]
            
            # Step 5: Manually compute softmax probabilities.
            # For numerical stability, subtract the max logit in each vocabulary slice.
            max_logits, _ = torch.max(shifted_logits, dim=-1, keepdim=True)
            stable_logits = shifted_logits - max_logits
            exp_logits = torch.exp(stable_logits)
            sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
            softmax_probs = exp_logits / sum_exp  # Shape: [batch, seq_len, vocab_size]
            
            # Step 6: Prepare valid mask and adjusted labels.
            # Create a mask of valid tokens (those not equal to ignore_index).
            valid_mask = (shifted_true_labels != ignore_index)
            # Create a copy of shifted_true_labels and replace ignore_index values with a dummy index (e.g., 0).
            adjusted_labels = shifted_true_labels.clone()
            adjusted_labels[~valid_mask] = 0
            
            # Step 7: Gather the probabilities for the true (shifted) labels.
            # Since adjusted_labels now contains valid indices everywhere, gather works without error.
            probs_for_true = softmax_probs.gather(dim=-1, index=adjusted_labels.unsqueeze(-1)).squeeze(-1)
            
            # Step 8: Compute token-level cross-entropy loss.
            epsilon = 1e-12
            token_losses = -torch.log(probs_for_true + epsilon)  # Shape: [batch, seq_len]
            # Zero out the losses for tokens that should be ignored.
            token_losses = token_losses * valid_mask.float()
            
            # Step 9: Average the loss only over valid tokens.
            total_loss = token_losses.sum()
            num_valid_tokens = valid_mask.sum().float()
            avg_loss = total_loss / num_valid_tokens
            
            # Step 10: Compute perplexity as the exponential of the average loss.
            perplexity = torch.exp(avg_loss).item()
            
        else:
            # Use the built-in loss from the model.
            with torch.no_grad():
                outputs = self.pp_model(input_ids, labels=input_ids)
                loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

    
    def calculate_bleu(self, references, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
        """
        Compute a sentence-level BLEU score from scratch.
        Uses the tutorial code with tokenization, modified precision with clipping,
        and applies a brevity penalty.
        """
        # If references is not a list, wrap it.
        if not isinstance(references, list):
            references = [references]
        # Define helper functions.
        def tokenize(sentence):
            return sentence.lower().split()

        def ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        def modified_precision(candidate_tokens, reference_tokens_list, n):
            candidate_ngrams = Counter(ngrams(candidate_tokens, n))
            max_ref_counts = Counter()
            for ref in reference_tokens_list:
                ref_ngrams = Counter(ngrams(ref, n))
                # For each ngram in candidate, update with maximum count observed among references.
                for ngram in candidate_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
            clipped_counts = {ngram: min(count, max_ref_counts[ngram])
                                for ngram, count in candidate_ngrams.items()}
            numerator = sum(clipped_counts.values())
            denominator = sum(candidate_ngrams.values())
            if denominator == 0:
                return 0
            return Fraction(numerator, denominator)

        def closest_reference_length(candidate_tokens, reference_tokens_list):
            candidate_len = len(candidate_tokens)
            ref_lens = [len(ref) for ref in reference_tokens_list]
            return min(ref_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))

        def brevity_penalty(candidate_tokens, reference_tokens_list):
            c_len = len(candidate_tokens)
            closest_len = closest_reference_length(candidate_tokens, reference_tokens_list)
            if c_len > closest_len:
                return 1
            else:
                return math.exp(1 - closest_len / c_len) if c_len > 0 else 0

        # Main BLEU computation.
        candidate_tokens = tokenize(candidate)
        reference_tokens_list = [tokenize(ref) for ref in references]
        precisions = []
        Hard_Smoothing = False
        for i in range(len(weights)):
            p = modified_precision(candidate_tokens, reference_tokens_list, i+1)
            if Hard_Smoothing:
                if p == 0:
                    p = Fraction(1, 10**9)  # smoothing: tiny value
            precisions.append(float(p))
        # Geometric mean of n-gram precisions.
        if all(p == 0 for p in precisions):
            return 0
        
        if Hard_Smoothing:
            geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
        else:
            geo_mean = math.exp(sum(w * math.log(float(p)) for w, p in zip(weights, precisions) if p != 0))
        bp = brevity_penalty(candidate_tokens, reference_tokens_list)
        bleu = bp * geo_mean
        return min(bleu, 1)

    def calculate_rouge(self, references, candidate):
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
        For each metric, compute scores for each reference and select the maximum F-measure.
        
        The parameter `references` can be either a single string or a list of strings.
        """
        # Ensure references is a list.
        if not isinstance(references, list):
            references = [references]
        
        def tokenize(sentence):
            return sentence.lower().split()
        
        def ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        def rouge_n(candidate_tokens, reference_tokens, n):
            cand_ngrams = Counter(ngrams(candidate_tokens, n))
            ref_ngrams = Counter(ngrams(reference_tokens, n))
            overlap = sum(min(cand_ngrams[ngram], ref_ngrams[ngram]) for ngram in cand_ngrams)
            precision = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0
            recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0
            if precision + recall == 0:
                fscore = 0
            else:
                fscore = 2 * precision * recall / (precision + recall)
            return precision, recall, fscore

        # Use the provided code for LCS.
        def lcs(text1, text2):
            n1 = len(text1)
            n2 = len(text2)
            dp = [[0] * (n1 + 1) for _ in range(n2 + 1)]
            for i in range(n2 - 1, -1, -1):
                for j in range(n1 - 1, -1, -1):
                    if text2[i] != text1[j]:
                        dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                    else:
                        dp[i][j] = 1 + dp[i + 1][j + 1]
            return dp[0][0]

        # Define ROUGE-L based on the LCS computed by our lcs function.
        def rouge_l(candidate_tokens, reference_tokens):
            lcs_length = lcs(candidate_tokens, reference_tokens)
            precision = lcs_length / len(candidate_tokens) if candidate_tokens else 0
            recall = lcs_length / len(reference_tokens) if reference_tokens else 0
            if precision + recall == 0:
                fscore = 0
            else:
                fscore = 2 * precision * recall / (precision + recall)
            return precision, recall, fscore

        # Tokenize candidate.
        candidate_tokens = tokenize(candidate)
        
        # Compute ROUGE scores for each reference and select the best for each metric.
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref in references:
            ref_tokens = tokenize(ref)
            p1, r1, f1 = rouge_n(candidate_tokens, ref_tokens, 1)
            p2, r2, f2 = rouge_n(candidate_tokens, ref_tokens, 2)
            pL, rL, fL = rouge_l(candidate_tokens, ref_tokens)
            rouge1_scores.append((p1, r1, f1))
            rouge2_scores.append((p2, r2, f2))
            rougeL_scores.append((pL, rL, fL))
        
        rouge1 = max(rouge1_scores, key=lambda x: x[2])
        rouge2 = max(rouge2_scores, key=lambda x: x[2])
        rougeL = max(rougeL_scores, key=lambda x: x[2])
        
        return {
            "rouge1": {"precision": rouge1[0], "recall": rouge1[1], "fmeasure": rouge1[2]},
            "rouge2": {"precision": rouge2[0], "recall": rouge2[1], "fmeasure": rouge2[2]},
            "rougeL": {"precision": rougeL[0], "recall": rougeL[1], "fmeasure": rougeL[2]}
        }
    
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
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            score = -1
            reasoning = "Evaluation error"

        # Compute additional deterministic metrics.
        perplexity = self.calculate_perplexity(response)
        bleu = self.calculate_bleu(reference, response)
        rouge_scores = self.calculate_rouge(reference, response)
        
        result = {
            "question": question,
            "reference": reference,
            "response": response,
            "score": score,
            "reasoning": reasoning[:200],
            "perplexity": perplexity,
            "bleu": bleu,
            "rouge1_precision": rouge_scores["rouge1"]["precision"],
            "rouge1_recall": rouge_scores["rouge1"]["recall"],
            "rouge1_fmeasure": rouge_scores["rouge1"]["fmeasure"],
            "rouge2_precision": rouge_scores["rouge2"]["precision"],
            "rouge2_recall": rouge_scores["rouge2"]["recall"],
            "rouge2_fmeasure": rouge_scores["rouge2"]["fmeasure"],
            "rougeL_precision": rouge_scores["rougeL"]["precision"],
            "rougeL_recall": rouge_scores["rougeL"]["recall"],
            "rougeL_fmeasure": rouge_scores["rougeL"]["fmeasure"],
        }

        return result

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
        print(f"Question: {result['question']}")
        print(f"Reference: {result['reference']}")
        print(f"Response: {result['response']}")
        print(f"LLM Judge Score: {result['score']}/5")
        print(f"LLM Judge Reasoning: {result['reasoning']}")
        print(f"Perplexity: {result['perplexity']}")
        print(f"BLEU: {result['bleu']}")
        print(f"ROUGE-1: Precision: {result['rouge1_precision']}, Recall: {result['rouge1_recall']}, F-measure: {result['rouge1_fmeasure']}")
        print(f"ROUGE-2: Precision: {result['rouge2_precision']}, Recall: {result['rouge2_recall']}, F-measure: {result['rouge2_fmeasure']}")
        print(f"ROUGE-L: Precision: {result['rougeL_precision']}, Recall: {result['rougeL_recall']}, F-measure: {result['rougeL_fmeasure']}")
        print("="*15)
    
    # Final evaluation report.
    if results:
        df = pd.DataFrame(results)
        print("\nEvaluation Report:")
        print(df[["question", "reference", "response", "score", "perplexity", "bleu", "rouge1_fmeasure"]].to_markdown(index=False))
        print(f"\nAverage LLM Judge Score: {df['score'].mean():.1f}/5")
        print(f"Average Perplexity: {df['perplexity'].mean():.2f}")
        print(f"Average BLEU: {df['bleu'].mean():.3f}")
        print(f"Average ROUGE-1 F-measure: {df['rouge1_fmeasure'].mean():.3f}")
    else:
        print("Error: No evaluation results generated")