#!/usr/bin/env python
# coding: utf-8

# In[3]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total no of characters:", len(raw_text))# it prints total no of characters in the book
print(raw_text[:99])# here prints the first 100 characters


# In[3]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total no of characters:", len(raw_text))# it prints total no of characters in the book
print(raw_text[:99])# here prints the first 100 characters


# In[3]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total no of characters:", len(raw_text))# it prints total no of characters in the book
print(raw_text[:99])# here prints the first 100 characters


# In[4]:


#our goal is to tokenize this 20480 -charcatres short story into individual words and special characters that we can turn into embeddings for llm training llm


# In[5]:


import re
#re.split(r'(\s'),text) here it splits the text based on white space point

# in both the splitsc the white spaces are treated as separate token 
text="hi, i am mounisree and im , learning llms."
result=re.split(r'(\s)',text)
print(result)


# In[6]:


#re.split(r'[,.]|\s',text) here it splits the text based on coma ,pullstop and space here to split the text we use regular expression
text="hi, i am mounisree and im , learning llms."
result=re.split(r'([,.]|\s)',text)
print(result)


# In[7]:


#here above we treat spaces as separate tokens so to remove spaces we simply use
# here item.strip()  helps to remove spaces here item.strip() return false if it is white space or else it returns true
#removing white spaces reduces the memory and computation requirements however keeping the white spaces can be useful 
#if we train models that are sensitive to the exact sensitive to the text ex:python-indentation so we need to check is it okay to remove whitespaces are not
text="hi, i am mounisree and im , learning llms."
result=re.split(r'([,.]|\s)',text)
result=[item for item in result if item.strip()]
print(result)


# In[8]:


text="hi? love u: it makes ? big -- pleasure"
result=re.split(r'([,.:;?_!"()\'] |--|\s)',text)
result=[item for item in result if item.strip()]
print(result);


# In[24]:


preprocessed=re.split(r'([,.:;?_!"()\'] |--|\s)',raw_text)
preprocessed=[item for item in preprocessed if item.strip()]
print(preprocessed[:30])
print("total tokens:",len(preprocessed))


# In[29]:


all_words=sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)


# In[26]:


vocab={token:Integer for Integer,token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break


# In[34]:


class SimpleTokenizer:
    def __init__(self,vaocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}
    def encode(self,text):
            preprocessed=re.split(r'([,.:;?_!"()\'] |--|\s)',text)
            preprocessed=[item for item in preprocessed if item.strip()]
            preprocessed=[
                item if item in self.str_to_int
                else "<|unk|>" for item in preprocessed
            ]
            ids=[self.str_to_int[s] for s in preprocessed]
            return ids
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text



# In[35]:


token=SimpleTokenizer(vocab)
text="hi im b mounisree "
ids=token.encode(text)
print(ids)


# In[36]:


all_words=sorted(set(preprocessed))
all_words.extend(["<|endoftext|>","<|unk|>"])
vocab={token:Integer for Integer,token in enumerate(all_words)}


# In[37]:


vocab.items()


# In[38]:





# In[ ]:




