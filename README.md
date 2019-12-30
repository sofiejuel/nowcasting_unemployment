## Beyond the Hype: A machine learning approach to macroeconomic nowcasting

Master's thesis, December 2019 <br/>
University of Copenhagen, Department of Economics <br/>
Sofie Juel and Waldemar Schoustrup Schuppli  

#### About 

This repository contains nearly all code, models and graphs produced for our master's thesis.  <br/>
The actual thesis can be found here: <a href="thesis/master_thesis.pdf" download="master_thesis.pdf">Master's Thesis</a>

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

#### Repository structure

[Thesis](thesis) contains the master's thesis as a PDF file
[Denmark](denmark) contains the data, code, models and results for Denmark
[Sweden](sweden)  contains the data, code, models and results for Sweden







