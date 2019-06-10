#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastai
from fastai.vision import *


# In[2]:


path = '../input/game-of-deep-learning-ship-datasets/train/'


# In[3]:


src = ImageList.from_csv(path,'train.csv', folder='images',
                   suffix='').split_by_rand_pct(0.2).\
                    label_from_df()


# In[4]:


tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[5]:


data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))


# In[6]:


data.show_batch(rows=3, figsize=(12,9))


# In[7]:


arch = models.resnet101


# In[8]:


from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


# In[9]:


# fbeta_binary = fbeta_binary_me(1)
learn = cnn_learner(data, arch,model_dir="/kaggle/working/")
learn.loss_fn = FocalLoss()


# In[10]:


learn.lr_find()


# In[11]:


learn.recorder.plot()


# In[12]:


lr = 0.01


# In[13]:


learn.fit_one_cycle(20, slice(lr))


# In[14]:


learn.save('stage-1-rn50')


# In[15]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[16]:


learn.unfreeze()


# In[17]:


learn.lr_find()
learn.recorder.plot()


# In[18]:


learn.fit_one_cycle(20, slice(1e-5, lr/5))


# In[19]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[20]:


learn.freeze()


# In[21]:


learn.lr_find()
learn.recorder.plot()


# In[22]:


lr=1e-2/2


# In[23]:


learn.fit_one_cycle(20, slice(lr))


# In[24]:


learn.save('stage-1-256-rn50')


# In[25]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[26]:


learn.unfreeze()


# In[27]:


learn.fit_one_cycle(30, slice(1e-5, lr/5))


# In[28]:


learn.recorder.plot_losses()


# In[29]:


learn.save('stage-2-256-rn50')


# In[30]:


learn.export('/kaggle/working/fastai_resnet.pkl')


# In[31]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[32]:


test = ImageList.from_folder('../input/avgameofdltestjpg/test-jpg/')
len(test)


# In[33]:



learn = load_learner('/kaggle/working/','fastai_resnet.pkl', test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[34]:


preds[:5]


# In[35]:


labelled_preds = torch.argmax(preds,1)+1


# In[36]:


# thresh = 0.2
# labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[37]:


fnames = [f.name for f in learn.data.test_ds.items]


# In[38]:


df = pd.DataFrame({'image':fnames, 'category':labelled_preds}, columns=['image', 'category'])


# In[39]:


df.to_csv('submission.csv', index=False)


# In[40]:


df.head()


# In[41]:


df.tail()


# In[42]:


df['class1_prob'] = preds[:,0].numpy()
df['class2_prob'] = preds[:,1].numpy()
df['class3_prob'] = preds[:,2].numpy()
df['class4_prob'] = preds[:,3].numpy()
df['class5_prob'] = preds[:,4].numpy()
df.to_csv('raw_prob.csv', index=False)

