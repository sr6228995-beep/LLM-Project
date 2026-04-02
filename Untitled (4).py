#!/usr/bin/env python
# coding: utf-8

# In[222]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total no of characters:", len(raw_text))# it prints total no of characters in the book
print(raw_text[:99])# here prints the first 100 characters


# In[223]:


get_ipython().system('pip install torch')


# In[224]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total no of characters:", len(raw_text))# it prints total no of characters in the book
print(raw_text[:99])# here prints the first 100 characters


# In[225]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total no of characters:", len(raw_text))# it prints total no of characters in the book
print(raw_text[:99])# here prints the first 100 characters


# In[226]:


#our goal is to tokenize this 20480 -charcatres short story into individual words and special characters that we can turn into embeddings for llm training llm


# In[227]:


import re
#re.split(r'(\s'),text) here it splits the text based on white space point

# in both the splitsc the white spaces are treated as separate token 
text="hi, i am mounisree and im , learning llms."
result=re.split(r'(\s)',text)
print(result)


# In[228]:


#re.split(r'[,.]|\s',text) here it splits the text based on coma ,pullstop and space here to split the text we use regular expression
text="hi, i am mounisree and im , learning llms."
result=re.split(r'([,.]|\s)',text)
print(result)


# In[229]:


#here above we treat spaces as separate tokens so to remove spaces we simply use
# here item.strip()  helps to remove spaces here item.strip() return false if it is white space or else it returns true
#removing white spaces reduces the memory and computation requirements however keeping the white spaces can be useful 
#if we train models that are sensitive to the exact sensitive to the text ex:python-indentation so we need to check is it okay to remove whitespaces are not
text="hi, i am mounisree and im , learning llms."
result=re.split(r'([,.]|\s)',text)
result=[item for item in result if item.strip()]
print(result)


# In[230]:


text="hi? love u: it makes ? big -- pleasure"
result=re.split(r'([,.:;?_!"()\'] |--|\s)',text)
result=[item for item in result if item.strip()]
print(result);


# In[231]:


preprocessed=re.split(r'([,.:;?_!"()\'] |--|\s)',raw_text)
preprocessed=[item for item in preprocessed if item.strip()]
print(preprocessed[:30])
print("total tokens:",len(preprocessed))


# In[232]:


all_words=sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)


# In[233]:


vocab={token:Integer for Integer,token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break


# In[234]:


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



# In[235]:


all_words=sorted(set(preprocessed))
all_words.extend(["<|endoftext|>","<|unk|>"])
vocab={token:Integer for Integer,token in enumerate(all_words)}


# In[236]:


token=SimpleTokenizer(vocab)
text="hi im b mounisree "
ids=token.encode(text)
print(ids)


# In[237]:


vocab.items()


# In[ ]:





# In[238]:


get_ipython().system('pip3 install tiktoken')


# In[239]:


import importlib
import tiktoken


# In[240]:


tokenizer=tiktoken.get_encoding("gpt2")


# In[241]:


text="Hello, do you like tea ? <|endoftext|> in the sunslit terraces of someunonowplace"
ids=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(ids)


# In[242]:


string=tokenizer.decode(ids)
print(string)


# In[243]:


#Creating Input-OutputTarget Pairs


# In[244]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

encode_txt=tokenizer.encode(raw_text)
print(len(encode_txt))


# In[245]:


enc_sample=encode_txt[50:]


# In[246]:


#here we are encoding the input text into tokens using byte pair encoding and then we are framing like input-output pairs


# In[289]:


context_size=5
x=enc_sample[:context_size]
y=enc_sample[1:context_size]
print("x:",x)
print("y:",y)
for i in range(1,context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    print(context,"----->",desired)


# In[290]:


for i in range(1,context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    print(context,"----->",desired)


# In[291]:


#everything in the left of arraow are input to the llm would receive and the token id on the right side refers the output predicting from the input


# In[292]:


for i in range(1,context_size+1):
    context=enc_sample[:i]
    desire=enc_sample[i]
    print(tokenizer.decode(context),"---->",tokenizer.decode([desire]))


# In[293]:


#we have now created the input-target pairs that can turn into use for llm training


# In[294]:


#there is only one more task before we can turn the tokens into embeddings:implementing an efficient data loader 
#that iterates over the input dataset and returns the inouts and targets as pytorch tensors which can thought of as multidimensional array 


# In[295]:


#in perticular we are interested in returning two tensors an input tensor containing the text that llms can see and the output tensor that llm predict 


# In[296]:


# Implementing Data Loader 


# In[297]:


#for efficient data loader implementation ,we will use pytorch's built in Dataset And Dataloader classes


# In[298]:


#step1:tokenize the entire text
#step2:use a sliding windo to chunk the book into overlapping sequences of maxlength
#step3:return total number of rows in the dataset
#step4: return single row from dataset


# In[299]:


get_ipython().system('pip install torch')


# In[300]:


from torch.utils.data import Dataset,DataLoader
class GPTDatasetv1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_id=[]
        self.target_id=[]
        token_ids=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_id.append(torch.tensor(input_chunk))
            self.target_id.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_id)
    def __getitem__(self,idx):
        return self.input_id[idx],self.target_id[idx]



# In[301]:


#the GPTDataSetv1 class in listing 2.5 is based on pytorch dataset class
#it defines how individual rows are fetched from the dataset
#each row consists of a number of token ids assaigned to input_chunk tensor
#the target_chunk contains the corresponding target


# In[302]:


#dataset is ready we will feed dataset to dataloader


# In[303]:


#step1:initialize the tokenizer
#step2:create dataset
#step3:drop_last=True drops the last batch if it is shorter than the special batch_size to prevent loss spies during training
#step4:the number of cpu processes to use for preprocessing 


# In[304]:


def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer=tiktoken.get_encoding("gpt2")
    dataset=GPTDatasetv1(txt,tokenizer,max_length,stride)#these method access the getitem() method in dataset
    dataloader=DataLoader(# batch size defines the no of batches that an model processes at once before updating its parameters
        dataset,#num_workers is for parellel processing like different threadas of cpu
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers

    )
    return dataloader


# In[305]:


with open("C:/Users/Mounisree/Downloads/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# In[306]:


import torch
dataloader=create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1,shuffle=False)
data_iter=iter(dataloader)
first_batch=next(data_iter)
print(first_batch)


# In[307]:


get_ipython().system('pip install torch')


# In[308]:


#stride defines ho many steps we need to move positions from input_field
# max_length=4 means each input ,output tensors contains 4 token ids


# In[309]:


second_batch=next(data_iter)
print(second_batch)


# In[310]:


#batch_size is small parameter update will be very quick but updates will be noisy
#batch_size is larger parameters update will be take long and updates will be less noisy
#just like regular deep learning ,the batch size is a trade off and hyperparameter to experiement with when training llms
#stride is large it prevents overfitting and overlapping 


# In[311]:


import torch
dataloader=create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4,shuffle=False)
data_iter=iter(dataloader)
first_batch=next(data_iter)
print(first_batch)


# In[312]:


input_ids=torch.tensor([2,3,5,1])


# In[313]:


vocab_size=6;
output_dim=3;
torch.manual_seed(123)
embedding_layer=torch.nn.Embedding(vocab_size,output_dim)


# In[314]:


print(embedding_layer.weight)


# In[315]:


print(embedding_layer(torch.tensor([3])))


# In[316]:


print(embedding_layer(input_ids))


# In[317]:


#positional Embedding 


# In[324]:


vocab_size=50257
output_dim=256
token_embedding_layer=torch.nn.Embedding(vocab_size,output_dim)


# In[325]:


max_length=4;
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False)
data_iter=iter(dataloader)
inputs,targets=next(data_iter)


# In[326]:


print("Token IDs:\n",inputs)
print("Input Shape:",inputs.shape)


# In[328]:


context_length=max_length
pos_embedding_layer=torch.nn.Embedding(context_length,output_dim)
pos_embedding=pos_embedding_layer(torch.arange(context_length))
print(pos_embedding.shape)


# In[332]:


# Token embedding
token_embeddings = token_embedding_layer(inputs)
pos_embedding = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embedding

print(input_embeddings.shape)   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




