# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

# # BERT
# 
# In this notebook we'll take a look at how we can use transformer models (like BERT) to create sentence vectors for calculating similarity. Let's start by defining a few example sentences.

# In[1]:
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

a = "purple is the best city in the forest"
b = "there is an art to getting your way and throwing bananas on to the street is not it"  # this is very similar to 'g'
c = "it is not often you find soggy bananas on the street"
d = "green should have smelled more tranquil but somehow it just tasted rotten"
e = "joyce enjoyed eating pancakes with ketchup"
f = "as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled"
g = "to get your way you must not bombard the road with yellow fruit"  # this is very similar to 'b'

# Installing dependencies needed for this notebook

# In[ ]:


# get_ipython().system('pip install -qU transformers sentence-transformers')


# In[ ]:


from transformers import AutoTokenizer, AutoModel
import torch

# Initialize our HF transformer model and tokenizer - using a pretrained SBERT model.

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
# save file to local
# tokenizer.save_pretrained(os.path.expanduser('~/Documents/Data/transformers_models/bert-base-nli-mean-tokens'))
# model.save_pretrained(os.path.expanduser('~/Documents/Data/transformers_models/bert-base-nli-mean-tokens'))

# Tokenize all of our sentences.

# In[ ]:


tokens = tokenizer([a, b, c, d, e, f, g],
                   max_length=128,
                   truncation=True,
                   padding='max_length',
                   return_tensors='pt')

# In[ ]:


print(tokens.keys())

# In[ ]:


print(tokens['input_ids'][0])

# Process our tokenized tensors through the model.

# In[ ]:

print(model)
outputs = model(**tokens)
print(outputs.keys())

# Here we can see the final embedding layer, *last_hidden_state*.

# In[ ]:


embeddings = outputs.last_hidden_state
print(embeddings[0])

# In[ ]:


print(embeddings[0].shape)

# Here we have our vectors of length *768*, but we see that these are not *sentence vectors* because we have a vector representation for each token in our sequence (128 in total). We need to perform a mean pooling operation to create the sentence vector.
# 
# The first thing we do is multiply each value in our `embeddings` tensor by its respective `attention_mask` value. The `attention_mask` contains **1s** where we have 'real tokens' (eg not padding tokens), and 0s elsewhere - so this operation allows us to ignore non-real tokens.

# In[ ]:


mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
print(mask.shape)

# In[ ]:


print(mask[0])

# Now we have a masking array that has an equal shape to our output `embeddings` - we multiply those together to apply the masking operation on our outputs.

# In[ ]:


masked_embeddings = embeddings * mask
print(masked_embeddings[0])

# Sum the remaining embeddings along axis 1 to get a total value in each of our 768 values.

# In[ ]:


summed = torch.sum(masked_embeddings, 1)
print(summed.shape)

# Next, we count the number of values that should be given attention in each position of the tensor (+1 for real tokens, +0 for non-real).

# In[ ]:


counted = torch.clamp(mask.sum(1), min=1e-9)
print(counted.shape)

# Finally, we get our mean-pooled values as the `summed` embeddings divided by the number of values that should be given attention, `counted`.

# In[ ]:


mean_pooled = summed / counted
print(mean_pooled.shape)

# Now we have our sentence vectors, we can calculate the cosine similarity between each.

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# In[ ]:


# convert to numpy array from torch tensor
mean_pooled = mean_pooled.detach().numpy()

# calculate similarities (will store in array)
scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
for i in range(mean_pooled.shape[0]):
    scores[i, :] = cosine_similarity(
        [mean_pooled[i]],
        mean_pooled
    )[0]

# In[ ]:


print(scores)

# We can visualize these scores using `matplotlib`.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# In[ ]:


plt.figure(figsize=(10, 9))
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
sns.heatmap(scores, xticklabels=labels, yticklabels=labels, annot=True)

# ---
# 
# ## Using sentence-transformers
# 
# The `sentence-transformers` library allows us to compress all of the above into just a few lines of code.

# In[ ]:

from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-large-nli-stsb-mean-tokens')
# model = AutoModel.from_pretrained('sentence-transformers/bert-large-nli-stsb-mean-tokens')


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

# We encode the sentences (producing our mean-pooled sentence embeddings) like so:

# In[ ]:


sentence_embeddings = model.encode([a, b, c, d, e, f, g])

# And calculate the cosine similarity just like before.

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# calculate similarities (will store in array)
scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
for i in range(sentence_embeddings.shape[0]):
    scores[i, :] = cosine_similarity(
        [sentence_embeddings[i]],
        sentence_embeddings
    )[0]

# In[ ]:


print(scores)

sentences = ['Lack of saneness',
             'Absence of sanity',
             'A man is eating food.',
             'A man is eating a piece of bread.',
             'The girl is carrying a baby.',
             'A man is riding a horse.',
             'A woman is playing violin.',
             'Two men pushed carts through the woods.',
             'A man is riding a white horse on an enclosed ground.',
             'A monkey is playing drums.',
             'A cheetah is running behind its prey.']
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

import scipy

query = 'Nobody has sane thoughts'  # A query sentence uses for searching semantic similarity score.
queries = [query]
query_embeddings = model.encode(queries)

print("Semantic Search Results")
number_top_matches = 3
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    print("Query:", query)
    print("\nTop {} most similar sentences in corpus:".format(number_top_matches))

    for idx, distance in results[0:number_top_matches]:
        print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1 - distance))


