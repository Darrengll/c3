#!/usr/bin/env python
# coding: utf-8

# # Parameter handling
# The tool within $C^3$ to manipulate the parameters of both the model and controls is the `ParameterMap`. It provides methods to present the same data for human interaction, i.e. structured information with physical units and for numerical optimization algorithms that prefer a linear vector of scale 1. Here, we'll show some example usage.
# We'll use the `ParameterMap` of the model also used in the simulated calibration example.

# In[1]:


from single_qubit_blackbox_exp import create_experiment

exp = create_experiment()
pmap = exp.pmap


# The pmap contains a list of all parameters and their values:

# In[2]:


pmap.get_full_params()


# To access a specific parameter, e.g. the frequency of qubit 1, we use the identifying tuple `('Q1','freq')`.

# In[3]:


pmap.get_parameter(('Q1','freq'))


# ## The opt_map
# To deal with multiple parameters we use the `opt_map`, a nested list of identifyers.

# In[4]:


opt_map = [
    [
        ("Q1", "freq")
    ],
    [
        ("Q1", "anhar")
    ],  
]


# Here, we get a list of the parameter values:

# In[5]:


pmap.get_parameters(opt_map)


# Let's look at the amplitude values of two gaussian control pulses, rotations about the $X$ and $Y$ axes repsectively.

# In[6]:


opt_map = [
    [
        ('rx90p[0]','d1','gauss','amp')
    ],
    [
        ('ry90p[0]','d1','gauss','amp')
    ],  
]


# In[7]:


pmap.get_parameters(opt_map)


# We can set the parameters to new values.

# In[8]:


pmap.set_parameters([0.5, 0.6], opt_map)


# In[9]:


pmap.get_parameters(opt_map)


# The opt_map also allows us to specify that two parameters should have identical values. Here, let's demand our $X$ and $Y$ rotations use the same amplitude.

# In[10]:


opt_map_ident = [
    [
        ('rx90p[0]','d1','gauss','amp'),
        ('ry90p[0]','d1','gauss','amp')
    ],
]


# The grouping here means that these parameters share their numerical value.

# In[11]:


pmap.set_parameters([0.432], opt_map_ident)
pmap.get_parameters(opt_map_ident)


# In[12]:


pmap.get_parameters(opt_map)


# During an optimization, the varied parameters do not change, so we fix the opt_map

# In[13]:


pmap.set_opt_map(opt_map)


# In[14]:


pmap.get_parameters()


# ## Optimizer scaling
# To be independent of the choice of numerical optimizer, they should use the methods

# In[15]:


pmap.get_parameters_scaled()


# To provide values bound to $[-1, 1]$. Let's set the parameters to their allowed minimum an maximum value with

# In[16]:


pmap.set_parameters_scaled([1.0,-1.0])


# In[17]:


pmap.get_parameters()


# As a safeguard, when setting values outside of the unit range, their physical values get looped back in the specified limits.

# In[18]:


pmap.set_parameters_scaled([2.0, 3.0])


# In[19]:


pmap.get_parameters()


# ## Storing and reading
# For optimization purposes, we can store and load parameter values in [HJSON](https://hjson.github.io/) format.

# In[20]:


pmap.store_values("current_vals.c3log")


# In[21]:


# !cat current_vals.c3log


# In[22]:


pmap.load_values("current_vals.c3log")



