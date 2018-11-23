#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium


# In[25]:


m = folium.Map(
    location=[31.169621, -99.683617],
    tiles='OpenStreetMap',
    zoom_start = 6
)


tooltip = 'Ciaone!'

folium.Marker([31.20, -99.80], popup='<i>Bella</i>', tooltip=tooltip).add_to(m)
folium.Marker([31.20, -98.80], popup='<b>a tutti!</b>', tooltip=tooltip).add_to(m)

m


# In[ ]:




