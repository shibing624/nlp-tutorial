#!/usr/bin/env python
# coding: utf-8

# # BERT
# 
# In this notebook we'll take a look at how we can use transformer models (like BERT) to create sentence vectors for calculating similarity. Let's start by defining a few example sentences.

# In[1]:
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
tokenizer.save_pretrained(os.path.expanduser('~/Documents/Data/transformers_models/bert-base-nli-mean-tokens'))
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

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
