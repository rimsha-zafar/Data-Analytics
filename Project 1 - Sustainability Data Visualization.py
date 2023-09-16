#!/usr/bin/env python
# coding: utf-8

# ### Import all the libraries

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ### Reading the sustainability dataset csv file

# In[2]:


data = pd.read_csv('WorldSustainabilityDataset.csv')


# In[3]:


data


# ### Renaming columns in a dataframe

# In[4]:


data = data.rename(columns={'Access to electricity (% of population) - EG.ELC.ACCS.ZS': '%population_with_electricity', 'Adjusted net national income per capita (annual % growth) - NY.ADJ.NNTY.PC.KD.ZG': '%GNIgrowth', 'Adjusted net savings, excluding particulate emission damage (% of GNI) - NY.ADJ.SVNX.GN.ZS':'%netemissionsavings', 'Adjusted savings: carbon dioxide damage (% of GNI) - NY.ADJ.DCO2.GN.ZS':'%CO2savings', 'Adjusted savings: natural resources depletion (% of GNI) - NY.ADJ.DRES.GN.ZS':'%NRD', 'Adjusted savings: particulate emission damage (% of GNI) - NY.ADJ.DPEM.GN.ZS':'%PPdamage', 'Adjusted savings: net forest depletion (% of GNI) - NY.ADJ.DFOR.GN.ZS':'%FD','Gini index (World Bank estimate) - SI.POV.GINI':'GINIindex', 'Income Classification (World Bank Definition)': 'Incomeclass', 'Individuals using the Internet (% of population) - IT.NET.USER.ZS':'%internetuse', 'Life expectancy at birth, total (years) - SP.DYN.LE00.IN':'LEB','Population, total - SP.POP.TOTL':'population', 'Rural population (% of total population) - SP.RUR.TOTL.ZS':'ruralpopulation', 'Regime Type (RoW Measure Definition)':'regime', 'Total natural resources rents (% of GDP) - NY.GDP.TOTL.RT.ZS':'NRrent','Urban population (% of total population) - SP.URB.TOTL.IN.ZS':'urbanpopulation','World Regions (UN SDG Definition)':'worldregions'})


# In[5]:


data


# In[6]:


data.describe()


# In[7]:


data.info()


# ### Check the category names and number of a categorical column using unique()

# In[8]:


data.worldregions.unique()


# In[9]:


data.Continent.unique()


# In[10]:


data.worldregions.nunique()


# ### Filtering a dataframe with a single and multiple conditions

# In[11]:


data[ (data['%CO2savings'] > 10) ]


# In[12]:


data[ (data['%CO2savings'] > 10) & (data['Incomeclass'] == 'Low income')  ]


# ### Select only the categorical columns

# In[13]:


data.select_dtypes(include= ['object'])


# ### Select only the numeric columns

# In[14]:


data.select_dtypes(exclude= ['object'])


# ### Groupby on a single column

# In[15]:


data.groupby('Country Name')['%GNIgrowth'].describe()


# ### Find correlation between two numerical columns using corr()

# In[16]:


data['%population_with_electricity'].corr(data['LEB'])


# ## Data Visualization

# ## Line Chart

# ### What is the trend of life expeectancy over the years?

# In[17]:


LEB_grouped = data.groupby('Year')['LEB'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=LEB_grouped.index, y=LEB_grouped.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Life Expectancy (years)')
plt.title('Trend of Life Expectancy Over Time')
plt.grid(True)
plt.show()


# ### What is the trend of GDP Per Capita Over Time?

# In[18]:


GDP_grouped = data.groupby('Year')['GDP per capita (current US$) - NY.GDP.PCAP.CD'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=GDP_grouped.index, y=GDP_grouped.values, marker='o')
plt.xlabel('Year')
plt.ylabel('GDP per Capita (current US$)')
plt.title('Trend of GDP per Capita Over Time')
plt.grid(True)
plt.show()


# ### How has the renewable electricity output changed as a percentage of total electricity output over the years?

# In[19]:


grouped_data5 = data.groupby('Year')['Renewable electricity output (% of total electricity output) - EG.ELC.RNEW.ZS'].mean()
plt.figure(figsize=(10, 6))
plt.plot(grouped_data5.index, grouped_data5.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Renewable Electricity Output (% of Total)')
plt.title('Trend of Renewable Electricity Output Over the Years')
plt.grid(True)
plt.show()


# ### What is the annual production of CO2 emissions by Continent over time?

# In[20]:


CO2_grouped = data.groupby(['Year', 'Continent'])['Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', hue='Continent', data=CO2_grouped)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (million tonnes)')
plt.title('CO2 Emissions by Continent Over Time')
plt.legend(title='Continent', loc='upper left')
plt.grid(True)
plt.show()


# ### How has the Renewable Energy consumption changed over time among different Income Classes?

# In[21]:


RE_grouped = data.groupby(['Year', 'Incomeclass'])['Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS', hue='Incomeclass', data=RE_grouped)
plt.xlabel('Year')
plt.ylabel('Renewable Energy Consumption (%)')
plt.title('Renewable Energy Consumption by Income Classification Over Time')
plt.legend(title='Income Classification', loc='upper left')
plt.grid(True)
plt.show()


# ### What are the gross savings over time by world regions?

# In[22]:


grosssaving_grouped = data.groupby(['Year', 'worldregions'])['Gross savings (% of GDP) - NY.GNS.ICTR.ZS'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Gross savings (% of GDP) - NY.GNS.ICTR.ZS', hue='worldregions', data=grosssaving_grouped)
plt.xlabel('Year')
plt.ylabel('Gross Savings (% of GDP)')
plt.title('Gross Savings Over Time by World Regions')
plt.legend(title='World Regions', loc='upper left')
plt.grid(True)
plt.show()


# ###  How has the unemployment rate changed for men and women over time?

# In[23]:


grouped_data2 = data.groupby('Year')['Unemployment rate, women (%) - SL_TLF_UEM - 8.5.2'].mean()
plt.plot(grouped_data2.index, grouped_data2.values, marker='o')
plt.xlabel('Year'); plt.ylabel('Average Unemployment Rate (Female)'); plt.title('Trend of Unemployment Rate (Female)')
plt.show()


# In[24]:


grouped_data3 = data.groupby('Year')['Unemployment rate, male (%) - SL_TLF_UEM - 8.5.2'].mean()
plt.plot(grouped_data3.index, grouped_data3.values, marker='o')
plt.xlabel('Year'); plt.ylabel('Average Unemployment Rate (Male)'); plt.title('Trend of Unemployment Rate (Male)')
plt.show()


# ### What is the trend of internet usage over the years across different continents?

# In[25]:


import seaborn as sns
grouped_data4 = data.groupby(['Year', 'Continent'])['%internetuse'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='%internetuse', hue='Continent', data=grouped_data4)
plt.xlabel('Year')
plt.ylabel('Internet Usage (% of population)')
plt.title('Trend of Internet Usage over Years by Continent')
plt.legend(title='Continent')
plt.show()


# ## Bar Chart

# ### What are the top 5 countries with highest production of CO2 emissions?

# In[26]:


country_emissions = data.groupby('Country Name')['Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes'].sum().reset_index()
top_emitters = country_emissions.sort_values(by='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', ascending=False)
top_5_emitters = top_emitters.head(5)
plt.figure(figsize=(12, 6))
sns.barplot(x='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', y='Country Name', data=top_5_emitters, palette='Reds_r')
plt.xlabel('CO2 Emissions (million tonnes)')
plt.ylabel('Country')
plt.title('Top 5 Countries with Highest CO2 Emissions')
plt.grid(True)
plt.show()


# ### What are the top 5 countries with lowest production of CO2 emissions?

# In[27]:


country_emissionss = data.groupby('Country Name')['Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes'].sum().reset_index()
lowest_emitters = country_emissionss.sort_values(by='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', ascending=True)
top_5_lowest_emitters = lowest_emitters.head(5)
plt.figure(figsize=(12, 6))
sns.barplot(x='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', y='Country Name', data=top_5_lowest_emitters, palette='Greens_r')
plt.xlabel('CO2 Emissions (million tonnes)')
plt.ylabel('Country')
plt.title('Top 5 Countries with Lowest CO2 Emissions')
plt.grid(True)
plt.show()


# ### What are the top 10 countries with the highest CO2 emissions in a 2017?

# In[28]:


year_data = data[data['Year'] == 2017]
top_countries =year_data.nlargest(10, 'Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes')
plt.figure(figsize=(12, 6))
sns.barplot(x='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', y='Country Name', data=top_countries)
plt.xlabel('CO2 Emissions (million tonnes)')
plt.ylabel('Country')
plt.title('Top Ten Countries with Highest CO2 Emissions in 2017')
plt.grid(True)
plt.show()


# ### What are the top 10 countries with the highest GDP in 2014?

# In[29]:


year_data2 = data[data['Year'] == 2014]
top_countries2 = year_data2.nlargest(10, 'GDP (current US$) - NY.GDP.MKTP.CD')
plt.figure(figsize=(12, 6))
sns.barplot(x='GDP (current US$) - NY.GDP.MKTP.CD', y='Country Name', data=top_countries2)
plt.xlabel('GDP (current US$)')
plt.ylabel('Country')
plt.title('Top Ten Countries with Highest GDP (current US$) in 2014')
plt.grid(True)
plt.show()


# ### What are the top 5 income classifications with the highest proportion of seats held by women in national parliaments in 2015?

# In[30]:


year_data4 = data[data['Year'] == 2015]
top_income_classifications = year_data4.nlargest(5, 'Proportion of seats held by women in national parliaments (%) - SG.GEN.PARL.ZS')
plt.figure(figsize=(10, 6))
sns.barplot(x='Proportion of seats held by women in national parliaments (%) - SG.GEN.PARL.ZS', y='Incomeclass', data=top_income_classifications)
plt.xlabel('Proportion of Seats Held by Women (%)')
plt.ylabel('Income Classification')
plt.title('Top Five Income Classifications with Highest Proportion of Seats Held by Women in 2015')
plt.grid(True)
plt.show()


# ### What is the GDP distribution among different income classifications over the years?

# In[53]:


grouped_dataa = data.groupby('Incomeclass')['GDP (current US$) - NY.GDP.MKTP.CD'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_dataa.index, y=grouped_dataa.values)
plt.xlabel('Income Classification')
plt.ylabel('Mean GDP (current US$)')
plt.title('Distribution of GDP among Income Classifications')
plt.xticks(rotation=0)
plt.show()


# ### How does the life expectancy vary among different continents over the years?

# In[32]:


grouped_dataaa = data.groupby('Continent')['LEB'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_dataaa.index, y=grouped_dataaa.values)
plt.xlabel('Continent')
plt.ylabel('Mean Life Expectancy (years)')
plt.title('Comparison of Life Expectancy among Continents')
plt.xticks(rotation=0)
plt.show()


# ## Scatter Plot

# ### How does internet usage correlate with GDP per capita?

# In[33]:


# Create a scatterplot
plt.figure(figsize=(10, 6))  

# Customize the plot 
plt.title('Correlation between Internet Usage and GDP per Capita')
plt.xlabel('Internet Usage (% of Population)')
plt.ylabel('GDP per Capita (current US$)')

# Plot the data points
plt.scatter(data['%internetuse'], data['GDP per capita (current US$) - NY.GDP.PCAP.CD'], alpha=0.6, c='blue', edgecolors='k')

# Add gridlines (optional)
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()


# ### How does internet usage correlate with GDP per capita in different income classes?

# In[34]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))  
plt.title('Correlation between Internet Usage and GDP per Capita by Income Class')
plt.xlabel('Internet Usage (% of Population)')
plt.ylabel('GDP per Capita (current US$)')
sns.scatterplot(x="%internetuse", y="GDP per capita (current US$) - NY.GDP.PCAP.CD", hue="Incomeclass", data=data, palette='Set1')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# ### How does Life Expectancy at Birth correlate with GDP per capita in different world regions?

# In[35]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))  
plt.title('Correlation between Life Expectancy and GDP per Capita by World Region')
plt.xlabel('Life Expectancy at Birth (years)')
plt.ylabel('GDP per Capita (current US$)')
sns.scatterplot(x="LEB", y="GDP per capita (current US$) - NY.GDP.PCAP.CD", hue="worldregions", data=data, palette='Set1')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='World Region')
plt.show()


# ### How does GDP per capita correlate with CO2 emissions in different regions?

# In[36]:


grouped_data6 = data.groupby('worldregions')[['GDP per capita (current US$) - NY.GDP.PCAP.CD', 'Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes']].mean()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GDP per capita (current US$) - NY.GDP.PCAP.CD', y='Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes', hue=grouped_data6.index, data=grouped_data6, s=120)
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')
plt.title('Correlation between GDP per Capita and CO2 Emissions by Region')
plt.legend(title='Region')
plt.show()


# ## Histogram

# ### What is the distribution of Life Expectancy at Birth?

# In[38]:


plt.hist(data['LEB'], bins=20)
plt.xlabel('Life Expectancy at Birth'); plt.ylabel('Frequency'); plt.title('Distribution of Life Expectancy')
plt.show()


# ## Box Plot

# ### How does the distribution of GINI Index vary among World Regions?

# In[40]:


import seaborn as sns
sns.boxplot(x= 'GINIindex', y='worldregions', data=data)
plt.gca().set_facecolor('#FFFF00')
plt.xlabel('GINI Index')
plt.ylabel('World Regions')
plt.title('Distribution of Gini Index by World Regions')


# ### Are there significant differences in the proportion of seats held by women in national parliaments among different world regions?

# In[41]:


sns.boxplot(x= 'worldregions', y='Proportion of seats held by women in national parliaments (%) - SG.GEN.PARL.ZS', data=data)
plt.gca().set_facecolor('#FFFF00')
plt.xlabel('World Regions')
plt.ylabel('Proportion of Seats Held by Women (%)')
plt.title('Distribution of Women\'s Representation in Parliaments by World Regions')
plt.xticks(rotation=90) 
plt.show()


# ## Heatmap

# In[37]:


dff = data[data['Country Name'] == 'Pakistan']
correlation_matrix = dff.corr()
plt.figure(figsize=(18, 12))  
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Heatmap of Pakistan')
plt.show()

