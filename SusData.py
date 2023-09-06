#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ### Reading the sustainability dataset csv file

# In[7]:


data = pd.read_csv('WorldSustainabilityDataset.csv')


# In[55]:


data


# ### Renaming columns in a dataframe

# In[12]:


data = data.rename(columns={'Access to electricity (% of population) - EG.ELC.ACCS.ZS': '%population_with_electricity', 'Adjusted net national income per capita (annual % growth) - NY.ADJ.NNTY.PC.KD.ZG': '%GNIgrowth', 'Adjusted net savings, excluding particulate emission damage (% of GNI) - NY.ADJ.SVNX.GN.ZS':'%netemissionsavings', 'Adjusted savings: carbon dioxide damage (% of GNI) - NY.ADJ.DCO2.GN.ZS':'%CO2savings', 'Adjusted savings: natural resources depletion (% of GNI) - NY.ADJ.DRES.GN.ZS':'%NRD', 'Adjusted savings: particulate emission damage (% of GNI) - NY.ADJ.DPEM.GN.ZS':'%PPdamage', 'Adjusted savings: net forest depletion (% of GNI) - NY.ADJ.DFOR.GN.ZS':'%FD','Gini index (World Bank estimate) - SI.POV.GINI':'GINIindex', 'Income Classification (World Bank Definition)': 'Incomeclass', 'Individuals using the Internet (% of population) - IT.NET.USER.ZS':'%internetuse', 'Life expectancy at birth, total (years) - SP.DYN.LE00.IN':'LEB','Population, total - SP.POP.TOTL':'population', 'Rural population (% of total population) - SP.RUR.TOTL.ZS':'ruralpopulation', 'Regime Type (RoW Measure Definition)':'regime', 'Total natural resources rents (% of GDP) - NY.GDP.TOTL.RT.ZS':'NRrent','Urban population (% of total population) - SP.URB.TOTL.IN.ZS':'urbanpopulation','World Regions (UN SDG Definition)':'worldregions'})


# In[13]:


data


# In[14]:


data.describe()


# In[15]:


data.info()


# ### Check the category names and number of a categorical column using unique()

# In[17]:


data.worldregions.unique()


# In[59]:


data.Continent.unique()


# In[18]:


data.worldregions.nunique()


# ### Filtering a dataframe with a single and multiple conditions

# In[20]:


data[ (data['%CO2savings'] > 10) ]


# In[21]:


data[ (data['%CO2savings'] > 10) & (data['Incomeclass'] == 'Low income')  ]


# ### Select only the categorical columns

# In[23]:


data.select_dtypes(include= ['object'])


# ### Select only the numeric columns

# In[25]:


data.select_dtypes(exclude= ['object'])


# ### Groupby on a single column

# In[27]:


data.groupby('Country Name')['%GNIgrowth'].describe()


# ### Find correlation between two numerical columns using corr()

# In[29]:


data['%population_with_electricity'].corr(data['LEB'])


# ## Data Visualization

# ## Bar Chart

# ### What is the average GNI distribution of different countries in Oceania region?

# In[30]:


df = data[data['worldregions'] == 'Oceania']
df


# In[108]:


import matplotlib.pyplot as plt
colors = ['red', 'green', 'blue', 'orange']
grouped_data = df.groupby('Country Name')['%GNIgrowth'].mean()
grouped_data.plot(kind='barh', color=colors)
plt.gca().set_facecolor('#AFEEEE')
plt.xlabel('Country Name'); plt.ylabel('Average GNI'); plt.title('Average GNI by Country')
plt.show()


# ### What is the GDP distribution among different income classifications?

# In[87]:


grouped_dataa = data.groupby('Incomeclass')['GDP (current US$) - NY.GDP.MKTP.CD'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_dataa.index, y=grouped_dataa.values)
plt.xlabel('Income Classification')
plt.ylabel('Mean GDP (current US$)')
plt.title('Distribution of GDP among Income Classifications')
plt.xticks(rotation=45)
plt.show()


# ### How does the life expectancy vary among different continents?

# In[88]:


grouped_dataaa = data.groupby('Continent')['LEB'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_dataaa.index, y=grouped_dataaa.values)
plt.xlabel('Continent')
plt.ylabel('Mean Life Expectancy (years)')
plt.title('Comparison of Life Expectancy among Continents')
plt.xticks(rotation=45)
plt.show()


# ## Line Chart

# ###  How has the unemployment rate changed for men and women over time?

# In[33]:


grouped_data2 = data.groupby('Year')['Unemployment rate, women (%) - SL_TLF_UEM - 8.5.2'].mean()
plt.plot(grouped_data2.index, grouped_data2.values)
plt.xlabel('Year'); plt.ylabel('Average Unemployment Rate (Female)'); plt.title('Trend of Unemployment Rate (Female)')
plt.show()


# In[34]:


grouped_data3 = data.groupby('Year')['Unemployment rate, male (%) - SL_TLF_UEM - 8.5.2'].mean()
plt.plot(grouped_data3.index, grouped_data3.values)
plt.xlabel('Year'); plt.ylabel('Average Unemployment Rate (Male)'); plt.title('Trend of Unemployment Rate (Male)')
plt.show()


# ### What is the trend of internet usage over the years across different continents?

# In[84]:


import seaborn as sns
grouped_data4 = data.groupby(['Year', 'Continent'])['%internetuse'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='%internetuse', hue='Continent', data=grouped_data4)
plt.xlabel('Year')
plt.ylabel('Internet Usage (% of population)')
plt.title('Trend of Internet Usage over Years by Continent')
plt.legend(title='Continent')
plt.show()


# ### How has the renewable electricity output changed as a percentage of total electricity output over the years?

# In[86]:


grouped_data5 = data.groupby('Year')['Renewable electricity output (% of total electricity output) - EG.ELC.RNEW.ZS'].mean()
plt.figure(figsize=(10, 6))
plt.plot(grouped_data5.index, grouped_data5.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Renewable Electricity Output (% of Total)')
plt.title('Trend of Renewable Electricity Output Over the Years')
plt.grid(True)
plt.show()


# ## Scatter Plot

# ### How does internet usage correlate with GDP per capita?

# In[35]:


plt.scatter(data['%internetuse'], data['GDP per capita (current US$) - NY.GDP.PCAP.CD'])
plt.xlabel('Internet Usage'); plt.ylabel('GDP per Capita'); plt.title('Internet Usage vs GDP per Capita')
plt.show()


# ### How does internet usage correlate with GDP per capita in different income classes?

# In[36]:


sns.relplot(x="%internetuse", y="GDP per capita (current US$) - NY.GDP.PCAP.CD", hue="Incomeclass", data=data);
plt.gca().set_facecolor('#90EE90')
plt.xlabel('Internet Usage')
plt.ylabel('GDP per Capita')


# ### How does Life Expectancy at Birth correlate with GDP per capita in different world regions?

# In[37]:


sns.relplot(x="LEB", y="GDP per capita (current US$) - NY.GDP.PCAP.CD", hue="worldregions", data=data);
plt.gca().set_facecolor('#90EE90')
plt.xlabel('Life Expectancy at Birth')
plt.ylabel('GDP per capita')


# ### How does GDP per capita correlate with CO2 emissions in different regions?

# In[85]:


grouped_data6 = data.groupby('worldregions')[['GDP per capita (current US$) - NY.GDP.PCAP.CD', 'Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes']].mean()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GDP per capita (current US$) - NY.GDP.PCAP.CD', y='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', hue=grouped_data6.index, data=grouped_data6)
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')
plt.title('Correlation between GDP per Capita and CO2 Emissions by Region')
plt.legend(title='Region')
plt.show()


# ## Heatmap

# In[101]:


dff = data[data['Country Name'] == 'Pakistan']
correlation_matrix = dff.corr()
plt.figure(figsize=(18, 12))  
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Heatmap of Pakistan')
plt.show()


# ## Histogram

# ### What is the distribution of Life Expectancy at Birth?

# In[39]:


plt.hist(data['LEB'], bins=20)
plt.xlabel('Life Expectancy at Birth'); plt.ylabel('Frequency'); plt.title('Distribution of Life Expectancy')
plt.show()


# ### How is the distribution of inflation rates among countries with different levels of trade as a percentage of GDP?

# In[80]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Inflation, consumer prices (annual %) - FP.CPI.TOTL.ZG', hue='Trade (% of GDP) - NE.TRD.GNFS.ZS', bins=20, multiple='stack')
plt.xlabel('Inflation Rate (annual %)')
plt.ylabel('Frequency')
plt.title('Distribution of Inflation Rates by Trade-to-GDP Ratio')
plt.legend(title='Trade-to-GDP Ratio')
plt.show()


# ## Box Plot

# ### How does the distribution of GINI Index vary among World Regions?

# In[46]:


import seaborn as sns
sns.boxplot(x= 'GINIindex', y='worldregions', data=data)
plt.gca().set_facecolor('#FFFF00')
plt.xlabel('GINI Index')
plt.ylabel('World Regions')
plt.title('Distribution of Gini Index by World Regions')


# ### Are there significant differences in the proportion of seats held by women in national parliaments among different world regions?

# In[75]:


sns.boxplot(x= 'worldregions', y='Proportion of seats held by women in national parliaments (%) - SG.GEN.PARL.ZS', data=data)
plt.gca().set_facecolor('#FFFF00')
plt.xlabel('World Regions')
plt.ylabel('Proportion of Seats Held by Women (%)')
plt.title('Distribution of Women\'s Representation in Parliaments by World Regions')
plt.xticks(rotation=45) 
plt.show()


# ### How does the distribution of access to electricity vary among countries with different income classifications?
# 

# In[105]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Incomeclass', y='%population_with_electricity', data=data)
plt.xlabel('Income Classification')
plt.ylabel('Access to Electricity (% of Population)')
plt.title('Distribution of Access to Electricity among Different Income Classifications')
plt.xticks(rotation=45)
plt.show()

