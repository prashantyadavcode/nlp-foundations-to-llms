# NLP Specific Metrics

# BLEU Score - used in - machine translation
# What BLEU Measures - compares generated translation with reference translation
# uses - N-gram overlap
# eg - 'the cat is on the mat' - prediction - 'the cat sits on the mat'  - BLEU checks overlap quality

# BLEU Formula (simplified) 
# 𝐵𝐿𝐸𝑈=𝐵𝑃×exp⁡(∑𝑤𝑛log⁡𝑝𝑛)

from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'sat', 'on', 'the', 'mat']

score = sentence_bleu(
    reference, candidate
)

print(score)


# ROUGE Score
# used in summarization

# ROUGE measures - overlap between generated summary, reference summary

# Types
# Metric - Meaning
# ROUGE-1  - unigram overlap
# ROUGE-2  - bigram overlap
# ROUGE-L  - longest common subsequences

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'roungeL'],
    use_stemmer = True
)

scores = scorer.socre(
    'The cat sat on the mat',
    'The cat is sitting on the mat'
)

print(scores)


