"""

"""

import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from flair.data import Sentence


import os

from repair.match_case import match_case_for_candidate


proxies = {"http": "Your_proxy_here", "https": "Your_proxy_here"}
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", proxies=proxies)
mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", proxies=proxies)


sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

LOGIT_WEIGHT = 0.2  #
SIM_WEIGHT = 0.8    #
TOP_K_CANDIDATES = 30  #
MIN_SIMILARITY_THRESHOLD = 0.3  #



#----------------------------------------------------------------------
"""special logics for MR1 repair"""
def repair_entity_mr1(
        original_sentence: str,
        mutated_sentence: str,       # 
        original_entity_text: str,
        mutated_entity_text: str,    # 
        original_tag: str,
        mutated_tag: str,
        tagger,
        model_type: str = "flair"
) -> Tuple[str, float]:

    print(f"\n===== Joint Repair (MR1): '{original_entity_text}' =====")

    scores_orig = _get_scores_for_sentence(original_sentence, original_entity_text, tagger, model_type, "Original")
    scores_mut = _get_scores_for_sentence(mutated_sentence, mutated_entity_text, tagger, model_type, "Mutated")

    final_scores = scores_orig.copy()
    for t, s in scores_mut.items():
        final_scores[t] = final_scores.get(t, 0.0) + s


    if original_tag in final_scores:
        print(f"  > MR1 Boost: Boosting {original_tag} score x 1.5")
        final_scores[original_tag] *= 1.5 
    

    best_tag = None
    best_score = -1
    for t, s in final_scores.items():
        if s > best_score:
            best_score = s
            best_tag = t

    if best_tag is None or best_score < 0.1:
        return original_tag, 1.0

    total = sum(final_scores.values())
    conf = best_score / total if total > 0 else 0.0
    
    print(f"  > Winner: {best_tag} ({conf:.2f})")
    return best_tag, conf



#-----------------------------------------------------------------------------------------

def find_entity_position(sentence: str, entity_text: str) -> Tuple[int, int]:
    """"""
    start = sentence.find(entity_text)
    if start == -1:
        #
        pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
        match = pattern.search(sentence)
        if match:
            start = match.start()
            end = match.end()
            return start, end
        return -1, -1

    end = start + len(entity_text)
    return start, end


def extract_context(sentence: str, entity_text: str, entity_start: int, entity_end: int) -> str:
    """"""
    before = sentence[:entity_start].strip()
    after = sentence[entity_end:].strip()

    if before and after:
        return f"{before} {after}"
    elif before:
        return before
    elif after:
        return after
    else:
        return ""


def generate_candidate_entities(context: str, entity_position: int, original_entity: str) -> List[Dict]:
    """ """
    # Insert mask token at entity position
    words = context.split()
    if entity_position >= len(words):
        entity_position = len(words) - 1

    words.insert(entity_position, tokenizer.mask_token)
    masked_text = " ".join(words)

    # Output masked sentence for debugging
    print(f"Masked sentence: '{masked_text}'")
    print(f"Mask token: '{tokenizer.mask_token}'")

    # Get predictions from BERT
    inputs = tokenizer(masked_text, return_tensors="pt")
    mask_positions = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    if len(mask_positions) == 0:
        # Handle case where mask token is tokenized differently
        print("Warning: Cannot find mask token position in input")
        return [{"text": original_entity, "logit": 1.0}]

    with torch.no_grad():
        outputs = mlm_model(**inputs)

    # Get top K predictions
    mask_position = mask_positions[0].item()
    logits = outputs.logits[0, mask_position, :]
    probs = torch.nn.functional.softmax(logits, dim=0)
    top_k = torch.topk(probs, TOP_K_CANDIDATES)

    print(f"Mask position: {mask_position}")

    candidates = []
    for i, (score, idx) in enumerate(zip(top_k.values, top_k.indices)):
        token = tokenizer.convert_ids_to_tokens(idx.item())
        # Remove special tokens and clean BERT wordpiece tokens
        if token.startswith("##"):
            token = token[2:]
        if token in tokenizer.all_special_tokens:
            continue

        # Match case
        matched_case_token = match_case_for_candidate(token, original_entity)

        candidates.append({
            "text": matched_case_token,
            "logit": score.item()
        })
        print(f"Candidate {i}: '{matched_case_token}' (original: '{token}'), probability: {score.item():.4f}")


    return candidates


def calculate_similarity_scores(context: str, candidates: List[Dict], entity_position: int) -> List[Dict]:
    """"""
    #
    context_embedding = sentence_model.encode(context)

    #
    words = context.split()

    for candidate in candidates:
        #
        candidate_words = words.copy()
        if entity_position < len(candidate_words):
            candidate_words.insert(entity_position, candidate["text"])
        else:
            candidate_words.append(candidate["text"])

        candidate_sentence = " ".join(candidate_words)

        #
        candidate_embedding = sentence_model.encode(candidate_sentence)

        #
        similarity = np.dot(context_embedding, candidate_embedding) / (
            np.linalg.norm(context_embedding) * np.linalg.norm(candidate_embedding)
        )

        candidate["similarity"] = float(similarity)

    return candidates



def predict_entity_tags(candidates: List[Dict], tagger, sentence: str, entity_start: int, entity_end: int,
                       model_type: str = "flair") -> List[Dict]:


   print("Predicting labels for each candidate entity...")

   for i, candidate in enumerate(candidates):
       candidate_text = candidate["text"]
       candidate_sentence = sentence[:entity_start] + candidate_text + sentence[entity_end:]
       print(f"  Candidate {i} ({candidate_text}): Replaced sentence '{candidate_sentence}'")

       # Get NER predictions
       flair_sentence = Sentence(candidate_sentence)
       tagger.predict(flair_sentence)

       # Choose different entity extraction methods based on model type
       entities = []
       if model_type.lower() == "flair":
           # Flair model uses get_spans method
           try:
               entities = flair_sentence.get_spans('ner')
               print(f"  Flair mode - Predicted entities: {[f'{e.text} ({e.tag})' for e in entities]}")
           except AttributeError as e:
               print(f"  Flair mode failed: {e}")
               # If Flair method fails, try cloud method as backup
               try:
                   entities = flair_sentence.spans.get('ner', [])
                   print(f"  Backup method - Entities from direct spans access: {[f'{e.text} ({e.tag})' for e in entities]}")
               except AttributeError:
                   print(f"  All methods failed, unable to extract entities")

       elif model_type.lower() == "cloud":
           # Azure/AWS models use spans attribute
           try:
               entities = flair_sentence.spans.get('ner', [])
               print(f"  Cloud mode - Entities from direct spans access: {[f'{e.text} ({e.tag})' for e in entities]}")
           except AttributeError:
               # If cloud mode fails, try Flair method as backup
               try:
                   entities = flair_sentence.get_spans('ner')
                   print(f"  Backup method - Entities from get_spans method: {[f'{e.text} ({e.tag})' for e in entities]}")
               except AttributeError as e:
                   print(f"  All methods failed: {e}")

       else:
           # Auto-detection mode - try Flair first, then Cloud
           try:
               entities = flair_sentence.get_spans('ner')
               print(f"  Auto-detection - Flair method successful: {[f'{e.text} ({e.tag})' for e in entities]}")
           except AttributeError:
               try:
                   entities = flair_sentence.spans.get('ner', [])
                   print(f"  Auto-detection - Cloud method successful: {[f'{e.text} ({e.tag})' for e in entities]}")
               except AttributeError as e:
                   print(f"  Auto-detection failed: {e}")

       # Find matching entity (logic is the same for both model types)
       candidate_entity = None

       # Method 1: Direct text matching
       for entity in entities:
           if hasattr(entity, 'text') and entity.text.lower() == candidate_text.lower():
               candidate_entity = entity
               print(f"  ✓ Found matching entity: {entity.text} ({entity.tag})")
               break

       # Method 2: Position matching as backup
       if not candidate_entity:
           candidate_start = entity_start
           candidate_end = entity_start + len(candidate_text)

           for entity in entities:
               if (hasattr(entity, 'start_position') and hasattr(entity, 'end_position') and
                       entity.start_position <= candidate_end and entity.end_position >= candidate_start):
                   candidate_entity = entity
                   print(f"  ✓ Found entity through position matching: {entity.text} ({entity.tag})")
                   break

       # Save predicted label
       if candidate_entity:
           candidate["predicted_tag"] = candidate_entity.tag
           print(f"  Set predicted label: {candidate_entity.tag}")
       else:
           candidate["predicted_tag"] = "O"
           print(f"  No matching entity found, set label to: O")

   return candidates



def compute_final_scores(candidates: List[Dict]) -> Dict[str, float]:
   """Calculate final scores for each entity type based on candidate entities"""
   tag_scores = {}
   print("Detailed calculation of scores for each candidate entity:")

   for i, candidate in enumerate(candidates):
       # Check if candidate entity has necessary fields
       if "similarity" not in candidate or "logit" not in candidate:
           print(f"  Candidate {i} missing similarity or logit: {candidate}")
           continue

       # Calculate combined score
       combined_score = (
               LOGIT_WEIGHT * candidate["logit"] +
               SIM_WEIGHT * candidate["similarity"]
       )

       # Only consider candidates with high enough similarity
       if candidate["similarity"] < MIN_SIMILARITY_THRESHOLD:
           print(
               f"  Candidate {i} ({candidate.get('text', 'UNKNOWN')}) similarity {candidate['similarity']} below threshold {MIN_SIMILARITY_THRESHOLD}, skipping")
           continue

       # Accumulate scores for predicted labels
       tag = candidate.get("predicted_tag", "O")
       print(
           f"  Candidate {i} ({candidate.get('text', 'UNKNOWN')}): similarity={candidate['similarity']:.4f}, logit={candidate['logit']:.4f}, "
           f"label={tag}, combined_score={combined_score:.4f}")

       if tag not in tag_scores:
           tag_scores[tag] = 0

       tag_scores[tag] += combined_score
       print(f"  Label {tag} accumulated score: {tag_scores[tag]:.4f}")

   print(f"Final label scores: {tag_scores}")
   return tag_scores


def _get_scores_for_sentence(sentence, entity_text, tagger, model_type, context_name):
    """Internal helper to get scores for a single sentence context"""
    print(f"  > Scoring context: {context_name}")
    # 1. Find position
    start, end = find_entity_position(sentence, entity_text)
    if start == -1:
        print(f"    Entity '{entity_text}' not found in {context_name}, skipping.")
        return {}
    
    # 2. Context & Masking
    ctx_str = extract_context(sentence, entity_text, start, end)
    words_before = sentence[:start].strip().split()
    pos = len(words_before)
    
    # 3. Candidates
    cands = generate_candidate_entities(ctx_str, pos, entity_text)
    
    # 4. Similarity
    cands = calculate_similarity_scores(ctx_str, cands, pos)
    
    # 5. Prediction
    cands = predict_entity_tags(cands, tagger, sentence, start, end, model_type)
    
    # 6. Scores
    return compute_final_scores(cands)


def repair_entity(
       original_sentence: str,
       mutated_sentence: str,      
       original_entity_text: str,  
       mutated_entity_text: str,   
       original_tag: str,
       mutated_tag: str,
       tagger,
       model_type: str = "flair"
) -> Tuple[str, float]:

   print(f"\n===== Joint Repair: '{original_entity_text}' =====")
   
   scores_orig = _get_scores_for_sentence(original_sentence, original_entity_text, tagger, model_type, "Original")
   

   scores_mut = _get_scores_for_sentence(mutated_sentence, mutated_entity_text, tagger, model_type, "Mutated")
   
   final_scores = scores_orig.copy()
   for t, s in scores_mut.items():
       final_scores[t] = final_scores.get(t, 0.0) + s
       
   print(f"  > Final Votes: {final_scores}")

   best_tag = None
   best_score = -1
   
   for t, s in final_scores.items():
       if s > best_score:
           best_score = s
           best_tag = t
           
   # Fallback
   if best_tag is None or best_score < 0.1:
       print(f"  > No winner. Keeping original tag: {original_tag}")
       return original_tag, 0.0
       
   # Confidence
   total = sum(final_scores.values())
   conf = best_score / total if total > 0 else 0.0
   
   print(f"  > Winner: {best_tag} ({conf:.2f})")
   return best_tag, conf

