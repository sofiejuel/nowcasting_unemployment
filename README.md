## Beyond the Hype: A machine learning approach to macroeconomic nowcasting

Master's thesis, December 2019 <br/>
University of Copenhagen, Department of Economics <br/>
Sofie Juel and Waldemar Schoustrup Schuppli


#### About 

This repository contains nearly all code, models and graphs produced for our master's thesis. Please cite us if you are inspired by our work. <br/>

We develop a machine learning approach to nowcasting regional unemployment rates that allows for real-time nowcasting within the publication lag period. We rely on novel data sources such as online search term intensity from Google Trends and job market indicators from Jobindex that are available in real time.
By combining various machine learning techniques with the novel data sources, we obtain
nowcasts for the monthly unemployment rates of the Danish regions from 2011-2019, and nowcasts for the quarterly unemployment rates of the Swedish regions from 2011- 2019. By testing various machine learning algorithms against different benchmark time series models for both countries, we find that the machine learning algorithms provides, at best, modest improvements the nowcasts of the unemployment rates – and we also analyse and discuss under which conditions machine learning has the most potential for improving nowcasts. <br/>

The actual thesis can be found here: <a href="thesis/master_thesis.pdf" download="master_thesis.pdf">Master's Thesis</a>. <br/>
The retrived data is available upon request. 

#### Repository structure

Note that the structure of the code is not made for general purposes, but are specific to our setup and analyis - thus, the code will most likely not run if one naively runs a given script. However, the code can serve as inspiration to anyone who wishes to conduct research that is similar to our thesis. 

<!-- Generate tree https://marketplace.visualstudio.com/items?itemName=Shinotatwu-DS.file-tree-generator -->

[denmark](denmark) contains the data, code, models and results for Denmark <br/>
 📦 denmark <br/>
 ┃ ┣ 📜1_merge_DK.ipynb <br/>
 ┃ ┣ 📜2_descriptive_DK.ipynb <br/>
 ┃ ┣ 📜3_analysis_DK.ipynb <br/>
 ┃ ┣ 📜4_model_weights_DK.ipynb <br/>
 ┃ ┣ 📜5_results_DK.ipynb <br/>
 ┃ ┗ 📜6_SHAP_xgboost_DK.ipynb <br/>

[data_fetch](data_fetch) contains the notebooks for retrieving data <br/> 
📦data_fetch  <br/>
 ┣ 📜DK_SE.ipynb  <br/>
 ┣ 📜control_variables.ipynb  <br/>
 ┣ 📜jobindex.ipynb  <br/>
 ┣ 📜pytrends.ipynb  <br/>
 ┗ 📜target_variable.ipynb  <br/>

[functions](functions) contains the constructed functions <br/> 
📦functions <br/>
 ┗ 📜func.py <br/>

[sweden](sweden)  contains the data, code, models and results for Sweden <br/> 
📦sweden <br/>
 ┃ ┣ 📜1_merge_robust.ipynb <br/>
 ┃ ┣ 📜2_descriptive_robust.ipynb <br/>
 ┃ ┣ 📜3_analysis_robust.ipynb <br/>
 ┃ ┣ 📜4_model_weights_robust.ipynb <br/>
 ┃ ┣ 📜5_analysis_conf_robust.ipynb <br/>
 ┃ ┗ 📜6_results_robust.ipynb <br/>

[thesis](thesis) contains the master's thesis as a PDF file <br/>
📦thesis <br/>
 ┗ 📜masters_thesis.pdf <br/>

#### Acknowledgements

There are a number of people whom we would like to offer our gratitude for their
helpful insights and comments throughout our entire thesis writing process.
First of all, we would like to thank Danmarks Nationalbank (The Central Bank of
Denmark) and the department of Data Analytics and Science for providing facilities
and guidance. In particular, Senior Data Scientist Alessandro Martinello has provided
support, insights and challenging questions through weekly feedback sessions during
the entirety of the thesis process, which has helped us tremendously in structuring
the analysis and sharpening our results.
Furthermore, we would like to thank our supervisor Andreas Bjerre-Nielsen for his
valuable feedback on how to motivate the analysis, and insights into how we should
illustrate the results. 

#### Abstract 

Many macroeconomic series such as unemployment rates are published with lags relative
to the period that the statistic covers, which renders decision-makers that rely on
these macroeconomic series blind for the duration of the publication lag period. Nowcasting
is an approach designed to alleviate this issue by providing an early estimate
of the desired statistic during the publication lag period.
In this thesis, we will develop a model framework to nowcast the regional unemployment
rates of Denmark and also expand the analysis to the regions of Sweden.
In order to this, we will incorporate two key aspects: Utilisation of alternative data
with very short publication lags, and machine learning techniques.
We utilise job posts from the largest online job market in Denmark, Jobindex,
along with search term intensity data from Google Trends. These data sources are
available almost real-time, which enables the creation of nowcasting models of the
regional unemployment rates during the publication lag period. By combining these
data sources with machine learning techniques, we capture more variation and interdependencies
in the data, which can allow for more accurate predictions.
Our findings suggest mixed results of applying novel real-time data combined
with machine learning techniques versus a traditional econometric nowcasting model.
The discovered improvements in nowcast precision of regional unemployment rates
are contingent on geography and the choice of the baseline model.




