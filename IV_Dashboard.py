#!/usr/bin/env python
# coding: utf-8

# # Cleaning and Preparing Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import country_converter as coco


import nltk

from plotly.subplots import make_subplots


# In[2]:


df = pd.read_csv('ds_salaries.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df = df.dropna()
df.info()


# ### Rename value to understand data better

# In[5]:


df['experience_level'] = df['experience_level'].replace({
    'SE': 'Senior-level', 'EN': 'Entry-level', 
    'EX': 'Executive-level', 'MI': 'Mid-level'})


# In[6]:


df['company_size'] = df['company_size'].replace({
    'S': 'Small', 'M': 'Medium', 'L' : 'Large'})


# In[7]:


df['employment_type'] = df['employment_type'].replace({
     'FT' : 'Full-time','PT' : 'Part-time',
     'FL': 'Freelancer','CT': 'Contractor'})


# In[8]:


df['remote_ratio'] = df['remote_ratio'].astype(str)
df['remote_ratio'] = df['remote_ratio'].replace({
    '0.0': 'On-Site', '50.0': 'Hybrid', '100.0' : 'Fully-Remote',
})


# In[9]:


df.head()


# # Visualizing Data
# ## Categorical features

# In[10]:


#group job title into six categories
def assign_broader_category(job_title):
    data_engineering = ["Data Engineer", "Data Analyst", "Analytics Engineer", "BI Data Analyst", "Business Data Analyst", "BI Developer", "BI Analyst", "Business Intelligence Engineer", "BI Data Engineer", "Power BI Developer"]
    data_scientist = ["Data Scientist", "Applied Scientist", "Research Scientist", "3D Computer Vision Researcher", "Deep Learning Researcher", "AI/Computer Vision Engineer"]
    machine_learning = ["Machine Learning Engineer", "ML Engineer", "Lead Machine Learning Engineer", "Principal Machine Learning Engineer"]
    data_architecture = ["Data Architect", "Big Data Architect", "Cloud Data Architect", "Principal Data Architect"]
    management = ["Data Science Manager", "Director of Data Science", "Head of Data Science", "Data Scientist Lead", "Head of Machine Learning", "Manager Data Management", "Data Analytics Manager"]
    
    if job_title in data_engineering:
        return "Data Engineering"
    elif job_title in data_scientist:
        return "Data Science"
    elif job_title in machine_learning:
        return "Machine Learning"
    elif job_title in data_architecture:
        return "Data Architecture"
    elif job_title in management:
        return "Management"
    else:
        return "Other"

# Apply the function to the 'job_title' column and create a new column 'job_category'
df['job_category'] = df['job_title'].apply(assign_broader_category)

df.head()


# In[11]:


salary = df['salary_in_usd']
fig3 = px.histogram(salary, x="salary_in_usd", nbins = 50, title = 'Salary Distribution')
fig3.update_layout(xaxis_title = "Salary (USD)", yaxis_title = "Count")
fig3.show()


# In[12]:


salary = df.sort_values(by=['work_year'], ascending=False)

work_year_lst =[2023, 2022, 2021, 2020]

fig5 = px.histogram(salary, x="salary_in_usd",  nbins = 50, title = 'Salary Distribution by years', animation_frame='work_year')

##Salary distribution (in USD)
#salary = df['salary_in_usd']
#fig = px.histogram(df, x="salary_in_usd", nbins = 50, title = 'Salary Distribution', animation_frame="work_year")
fig5.update_layout(xaxis_title = "Salary (USD)", yaxis_title = "Count")
#fig5["layout"].pop("updatemenus") 
fig5.show()


# In[13]:


salary_loc = df.groupby(['salary_in_usd', 'company_location']).size().reset_index()
salary_mean = salary_loc.groupby('company_location').mean().reset_index()

fig4 = px.treemap(salary_mean, path = [salary_mean['company_location']], values = salary_mean['salary_in_usd'],
                labels = {"labels": "Country", "salary_in_usd": "Average Salary (USD)"}, title = "Average salary in each country")
fig4.update_traces(textinfo = 'label +  value', texttemplate='%{label: labels } <br>$%{value} ')
fig4.show()


# In[14]:


fig6 = go.Figure()


fig6.add_trace(go.Box(x = df['experience_level'], y = df['salary_in_usd'], name="Experience level"))

fig6.add_trace(go.Box(x = df['company_size'], y = df['salary_in_usd'], name="Company Size"))

fig6.add_trace(go.Box(x = df['job_category'], y = df['salary_in_usd'], name="Job Category"))

fig6.add_trace(go.Box(x = df["employment_type"], y = df["salary_in_usd"], name="Employment type"))

fig6.add_trace(go.Box(x = df['remote_ratio'], y = df['salary_in_usd'], name="Remote Ratio"))

fig6.update_xaxes(categoryorder='array', categoryarray= ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level', 'Small', 'Medium', 'Large'])

fig6.layout.update(title = "Salary distibution based on different variables", updatemenus = [
    go.layout.Updatemenu(
        type = "buttons", direction = "down", active = 0, x = -0.1, y = 1,
        buttons = list(
            [
               dict(
                    label = "Experience level", method = "update",
                    args = [{"visible": [True, False, False, False, False]},{"title": "Salary Distribution based on Employment Type"} ]
                ),
               dict(
                    label = "Company Size", method = "update", 
                    args = [{"visible": [False, True, False, False, False]},{"title": "Salary Distribution based on Experience Level"}]
               ),
               dict(
                    label = "Job Category", method = "update",
                    args = [{"visible": [False, False, True, False, False]},{"title": "Salary Distribution based on Employment Type"} ]
                ),
               dict(
                    label = "Emmployment type", method = "update",
                    args = [{"visible": [False, False, False, True, False]},{"title": "Salary Distribution based on Employment Type"} ]
                ),
               dict(
                    label = "Remote Ratio", method = "update",
                    args = [{"visible": [False, False, False, False, True]},{"title": "Salary Distribution based on Employment Type"} ]
                ),
                dict(
                    label = "Back to main", method = "update",
                    args = [{"visible": [True, True, True, True, True]},{"title": "Salary Distribution based on different variables"} ]
                ),
                
            ]
        )
    )
])

fig6.show()



# In[15]:


#df_group = px.df.tips()
year_2020 = df[df["work_year"].isin([2020,2020])]
year_2021 = df[df["work_year"].isin([2021,2021])]
year_2022 = df[df["work_year"].isin([2022,2022])]
year_2023 = df[df["work_year"].isin([2023,2023])]

job_2020 = year_2020['job_category'].value_counts()
job_2021 = year_2021['job_category'].value_counts()
job_2022 = year_2022['job_category'].value_counts()
job_2023 = year_2023['job_category'].value_counts()

x= ['Data Engineering', 'Data Science', 'Machine Learning', 'Data Architecture', 'Management', 'Other']

fig7 = go.Figure(data=[go.Bar(
    name = '2020',
    x = x,
    y = job_2020.values,
    marker_color= 'red'
   ),
                       go.Bar(
    name = '2021',
    x = x,
    y = job_2021.values,
    marker_color= 'blue'
   ),
                       go.Bar(
    name = '2022',
    x = x,
    y = job_2022.values,
    marker_color= 'green'
   ),
                       go.Bar(
    name = '2023',
    x = x,
    y = job_2023.values,
    marker_color= 'orange'
   )
])

fig7.layout.update(title = "Number of employees based on job category",
   updatemenus = [
      go.layout.Updatemenu(
         type = "dropdown", direction = "down", active = 0, x = -0.1, y = 1.0,
         buttons = list(
            [
               dict(
                  label = "All years", method = "update",
                  args = [{"visible": [True, True, True, True]},{"title": "Number of employees in all years"} ]
               ),
               dict(
                  label = "2020", method = "update",
                  args = [{"visible": [True, False, False, False]},{"title": "Number of employees in 2020"} ]
               ),
               dict(
                  label = "2021", method = "update",
                  args = [{"visible": [False, True, False, False]},{"title": "Number of employees in 2021"} ]
               ),
               dict(
                  label = "2022", method = "update",
                  args = [{"visible": [False, False, True, False]},{"title": "Number of employees in 2022"} ]
               ),
               dict(
                  label = "2023", method = "update",
                  args = [{"visible": [False, False, False, True]},{"title": "Number of employees in 2023"} ]
               ) 
            ]
         )
      )
   ]
)

fig7.show()


# In[16]:


#fig8 = go.Figure()

fig8 = make_subplots(rows=1, cols=3)

ex_level = df['experience_level'].value_counts()

fig8.add_trace(go.Pie(labels = ex_level.index, values = ex_level.values, 
                      texttemplate = "%{label} <br>%{percent}"))

com_size = df['company_size'].value_counts()

fig8.add_trace(go.Pie(labels= com_size.index, values= com_size.values,
                      texttemplate = "%{label} <br>%{percent}"))

emp_type = df['employment_type'].value_counts()

fig8.add_trace(go.Pie(labels = emp_type.index, values = emp_type.values,
                      texttemplate = "%{label} <br>%{percent}"))

rem_rate = df['remote_ratio'].value_counts()

fig8.add_trace(go.Pie(labels = rem_rate.index, values = rem_rate.values,
                      texttemplate = "%{label} <br>%{percent}"))

work_year = df['work_year'].value_counts()

fig8.add_trace(go.Pie(labels = work_year.index, values = work_year.values,
                      texttemplate = "%{label} <br>%{percent}"))

fig8.layout.update(title ="About the data set",
   updatemenus = [
      go.layout.Updatemenu(
             type = "buttons", direction = "down", active = 0, x = -0.3, y = 1,
         buttons = list(
            [
               dict(
                  label = "Experience Level", method = "update",
                  args = [{"visible": [True, False, False, False, False]},{"title": "About the data set - Experience Level"} ]
               ),
               dict(
                  label = "Company Size", method = "update", 
                  args = [{"visible": [False, True, False, False, False]},{"title": "About the data set - Company Size"}]
               ),
               dict(
                  label = "Employment Type", method = "update", 
                  args = [{"visible": [False, False, True, False, False]},{"title": "About the data set - Employment Type"}]
               ), 
               dict(
                  label = "Remote Ratio", method = "update", 
                  args = [{"visible": [False, False, False, True, False]},{"title": "About the data set - Remote Ratio"}]
               ),
               dict(
                  label = "Work Year", method = "update", 
                  args = [{"visible": [False, False, False, False, True]},{"title": "About the data set - Work Year"}]
               ),
               
            ]
         )
      )
   ]
)

fig8.show()


# In[17]:


##Company location
country = coco.convert(names = df['company_location'], to = "ISO3")
df_loc = df
df_loc['company_location'] = country
com_loc = df_loc['company_location'].value_counts()

#employee residence
country = coco.convert(names = df['employee_residence'], to = "ISO3")
df_res = df
df_res['employee_residence'] = country
emp_res = df_res['employee_residence'].value_counts()

#average salary
emp_salary = df[['experience_level','salary_in_usd']]
salary_loc = df.groupby(['salary_in_usd', 'company_location']).size().reset_index()
salary_mean = salary_loc.groupby('company_location').mean().reset_index()

fig_t = px.choropleth(locations = com_loc.index,
                    color = com_loc.values,
                    color_continuous_scale=px.colors.sequential.GnBu,
                    title = 'Map', labels={ 'color': 'Number of employee/<br>Average Salary(USD)', 'locations': 'Country'})

button1 =  dict(method = "restyle",
                args = [{'z': [ com_loc.values ] },  {"title": "Company Location Distribution"}],
                label = "Company Location")
button2 =  dict(method = "restyle",
                args = [{'z': [  emp_res.values ]},  {"title": "Employee Residence Distribution"}],
                label = "Employee Residence")
button3 =  dict(method = "restyle",
                args = [{'z': [  salary_mean['salary_in_usd'] ]},  { "title": "Average Salary Distribution"}],
                label = "Average salary")



fig_t.update_layout(width=1000, title ="Map",
                  coloraxis_colorbar_thickness=23,
                  updatemenus=[dict(y=1.0,
                                    x=-0.1,
                                    xanchor='right',
                                    yanchor='top',
                                    active=0,
                                    buttons=[button1, button2, button3])
                              ]) 
fig_t.show()


# In[18]:


emp_salary = df[['experience_level','salary_in_usd']]

#'SE': 'Senior-level', 'EN': 'Entry-level', 'EX': 'Executive-level', 'MI': 'Mid-level'})

salary_loc = df.groupby(['salary_in_usd', 'company_location']).size().reset_index()
salary_mean = salary_loc.groupby('company_location').mean().reset_index()
fig9 = px.choropleth(locations = salary_mean['company_location'], color = salary_mean['salary_in_usd'],
                    title = 'Average Salary Distribution based on Company Location', 
                    labels={'color': 'Average Salary', 'locations': 'Country'})
fig9.show()


# In[19]:


fig10 = px.scatter(df, x="company_location", y="employee_residence", color='salary_in_usd', size='salary_in_usd', 
                   title = "Salary comparison between employee residence and company location",
                   labels = {'company_location': 'Company Location', 'employee_residence': 'Employee Residence', 'salary_in_usd': 'Salary (USD)'}
                   )
fig10.show()


# In[20]:


#fig10 = px.scatter(df, x="company_location", y="company_size", color='salary_in_usd', size='salary_in_usd', 
#                  title = "Salary comparison between employee residence and company location",
#                   labels = {'company_location': 'Company Location', 'employee_residence': 'Employee Residence', 'salary_in_usd': 'Salary (USD)'}
#                   )
#fig10.show()


#fig10 = go.Figure()

#fig10.add_trace(go.Scatter(x = df["company_location"], y = df["company_size"], z = df["salary_in_usd"]))

#fig10.show()


# In[21]:


top20 = df['job_title'].value_counts()[:20]
fig11 = px.bar(y = top20.values, x = top20.index,  title = 'The Top 20 Most Popular Jobs')
fig11.update_layout(xaxis_title = "Job Title", yaxis_title = "Count")
fig11.show()


# # Dashboard

# In[22]:


import dash
# html is used to set up the layout, and dcc is used to embed the graphs to the dashboard:
from dash import dcc
from dash import html

from jupyter_dash import JupyterDash

import dash_bootstrap_components as dbc

# In[23]:


# Setup the style from the link:
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.LITERA]
# Embed the style to the dashabord:
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
server = app.server



# In[24]:


graph1 = dcc.Graph(
        id='graph1',
        figure=fig11,
        className="six columns" 
    )
graph2 = dcc.Graph(
        id='graph2',
        figure=fig7,
        className="six columns"
    )
graph3 = dcc.Graph(
        id='graph3',
        figure=fig9,
        className="six columns"
    )
graph4 = dcc.Graph(
        id='graph4',
        figure=fig4,
        className="six columns"
    )
graph5 = dcc.Graph(
        id='graph5',
        figure=fig10,
        className="twelve columns"
    )
graph6 = dcc.Graph(
        id='graph6',
        figure=fig5,
        className="four columns"
    )
graph7 = dcc.Graph(
        id='graph7',
        figure=fig6,
        className="twelve columns"
    )
graph8 = dcc.Graph(
        id='graph8',
        figure=fig8,
        className="twelve columns"
    )
graph9 = dcc.Graph(
        id='graph9',
        figure=fig3,
        className="eight columns"
    )


# In[25]:


# setup the header
icon = html.H1(children="📈", style={'fontSize': "50px"})
title = html.H1(children="Salary of Data Science Overview", style={'fontSize': "60px", 'color': '#357EC3'})
desc1 = html.H4(children="This dashboard provides analysis of the dataset: Salaries of Different Data Science Fields in the Data Science Domain", style={'color': '#616D7E'})
#616D7E
header1 = html.Div(children =[icon, title, desc1], className="header",style={'backgroundColor':'#DBE9FA'})

row7 = html.Div(children=[graph9, graph6])
row4 = html.Div(children=[graph7])

#Location

gap = html.H2(children = "•")
arrow = html.H2(children = "🡻")
icon2 = html.H1(children="🗺️", style={'fontSize': "40px"})
desc2 = html.H2(children = "How location affects the salary?",  style={'fontSize': "30px", 'color': '#357EC3'})
desc2_extra = html.H4(children = "Hover or click on data to see more information.", style={'color': '#616D7E'})

row2 = html.Div(children=[graph3, graph4])
row3 = html.Div(children=[graph5])

desc2_extra1 = html.H4(children = "*Drag to zoom in and double click to zoom out the plot.", style={'color': '#616D7E',"text-align": "left"})

#part1 = html.Div(children = [desc2, row2, row3])
divider = html.Div(children = [gap, gap, gap, arrow], style={ 'color': '#357EC3'})
part1 = html.Div(children = [icon2, desc2, desc2_extra])
#Job titles
icon3 = html.H1(children="👜", style={'fontSize': "40px"})
desc3 = html.H2(children = "What are the most famous jobs in Data Science field?",   style={'fontSize': "30px", 'color': '#357EC3'})
row1 = html.Div(children=[graph1, graph2])
#part2 = html.Div(children = [gap, gap, gap, arrow, desc3, row1])
part2 = html.Div(children = [icon3, desc3])

#Data Info
icon4 = html.H1(children="📋", style={'fontSize': "40px"})
desc4 = html.H2(children = "More about the data set...",  style={'fontSize': "30px", 'color': '#357EC3'})
desc5 = html.H4(children = "Click on different variables to see the distribution on employees in this data set based on the variables selected.", style={'color': '#616D7E'})
row5 = html.Div(children=[graph8])
part3 = html.Div(children =[icon4, desc4, desc5])
#part3 = html.Div(children =[gap, gap, gap, arrow, desc4, row5])





#border = html.p(children="dotted")
# setup to rows, graph 1-3 in the first row, and graph4 in the second:
#header = html.Div(children =[title, desc1, line])

#row7 = html.Div(children=[graph9, graph6])
#row4 = html.Div(children=[graph7])
#row1 = html.Div(children=[graph1, graph2])
#row2 = html.Div(children=[graph3, graph4])

#row3 = html.Div(children=[graph5])
#row4 = html.Div(children=[graph7])
#row5 = html.Div(children=[graph8])

# setup & apply the layout
#layout = html.Div(children=[title, desc1, line, row7, row4, part1, part2, part3], style={"text-align": "center"})
layout = html.Div(children=[header1, row7, row4, divider, part1, row2, row3, desc2_extra1, divider, part2, row1, divider, part3, row5] , style={"text-align": "center"})
app.layout = layout
app.title ="Salary of Data Science Field"


# In[26]:


if __name__ == "__main__":
    app.run_server()

