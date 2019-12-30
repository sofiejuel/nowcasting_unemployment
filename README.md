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

[Thesis](thesis) contains the master's thesis as a PDF file <br/>
[Denmark](denmark) contains the data, code, models and results for Denmark <br/>
[Sweden](sweden)  contains the data, code, models and results for Sweden <br/>

Generate tree https://marketplace.visualstudio.com/items?itemName=Shinotatwu-DS.file-tree-generator



📦sweden
 ┣ 📂data
 ┃ ┣ 📂controls
 ┃ ┃ ┣ 📜df_DK_controls.csv
 ┃ ┃ ┗ 📜df_SE_controls.csv
 ┃ ┣ 📂descriptive
 ┃ ┃ ┣ 📜df_DK_descriptive.csv
 ┃ ┃ ┣ 📜df_analysis.csv
 ┃ ┃ ┗ 📜df_descriptive.csv
 ┃ ┣ 📂gt
 ┃ ┃ ┗ 📜dfTrends.csv
 ┃ ┣ 📂job_posts
 ┃ ┃ ┣ 📜df_DK_jobposts_quarterly_final.csv
 ┃ ┃ ┣ 📜df_DK_labour_force.csv
 ┃ ┃ ┣ 📜df_SE_jobposts_quarterly_final.csv
 ┃ ┃ ┗ 📜df_SE_labour_force.csv
 ┃ ┗ 📂target
 ┃ ┃ ┣ 📜df_DK_target.csv
 ┃ ┃ ┗ 📜df_SE_target.csv
 ┣ 📂report
 ┃ ┣ 📂8_robustness_DK
 ┃ ┃ ┗ 📜8_score_relative_DK.pdf
 ┃ ┣ 📂8_robustness_SE
 ┃ ┃ ┣ 📜.DS_Store
 ┃ ┃ ┣ 📜8_1stdiff_dist.pdf
 ┃ ┃ ┣ 📜8_conf.pdf
 ┃ ┃ ┣ 📜8_error_bar.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Blekinge_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Dalarna_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Gavleborg_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Gotland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Halland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Jonkoping_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Kalmar_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Kronoberg_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Norrbotten_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Orebro_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Ostergotland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Skane_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Sodermanland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Stockholm_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Uppsala_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Varmland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Vasterbotten_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Vasterbotten_xgboost_1.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Vasternorrland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Vastmanland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_Vastra_Gotalands_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_diff_stockholm_weighted_v_baseline.pdf
 ┃ ┃ ┣ 📜8_pred_diff_stockholm_xgboost_v_baseline.pdf
 ┃ ┃ ┣ 📜8_pred_level_Blekinge_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Dalarna_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Gavleborg_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Gotland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Halland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Jonkoping_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Kalmar_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Kronoberg_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Norrbotten_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Orebro_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Ostergotland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Skane_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Sodermanland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Stockholm_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Uppsala_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Varmland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Vasterbotten_baseline.pdf
 ┃ ┃ ┣ 📜8_pred_level_Vasterbotten_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Vasterbotten_xgboost_1.pdf
 ┃ ┃ ┣ 📜8_pred_level_Vastmanland_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Vastra_Gotalands_xgboost.pdf
 ┃ ┃ ┣ 📜8_pred_level_Västernorrland_xgboost.pdf
 ┃ ┃ ┣ 📜8_regional_boxplot_weighted.pdf
 ┃ ┃ ┣ 📜8_regional_boxplot_xgboost.pdf
 ┃ ┃ ┣ 📜8_regional_gain.pdf
 ┃ ┃ ┗ 📜8_score_relative.pdf
 ┃ ┗ 📜.DS_Store
 ┣ 📂results
 ┃ ┣ 📂final
 ┃ ┃ ┣ 📂baseline
 ┃ ┃ ┃ ┣ 📜results_ar1.pickle
 ┃ ┃ ┃ ┗ 📜results_ar_year_lag.pickle
 ┃ ┃ ┣ 📂bootstrap
 ┃ ┃ ┃ ┗ 📜results_bootstrap.pickle
 ┃ ┃ ┣ 📂elastic
 ┃ ┃ ┃ ┣ 📜results_final.pickle
 ┃ ┃ ┃ ┗ 📜results_mp.pickle
 ┃ ┃ ┣ 📂lasso
 ┃ ┃ ┃ ┣ 📜results_final.pickle
 ┃ ┃ ┃ ┗ 📜results_mp.pickle
 ┃ ┃ ┣ 📂randomforest
 ┃ ┃ ┃ ┣ 📜results_final.pickle
 ┃ ┃ ┃ ┣ 📜results_final_noint.pickle
 ┃ ┃ ┃ ┣ 📜results_mp.pickle
 ┃ ┃ ┃ ┗ 📜results_noint.pickle
 ┃ ┃ ┣ 📂ridge
 ┃ ┃ ┃ ┣ 📜results_final.pickle
 ┃ ┃ ┃ ┗ 📜results_mp.pickle
 ┃ ┃ ┣ 📂weighted
 ┃ ┃ ┃ ┣ 📜results_final.pickle
 ┃ ┃ ┃ ┗ 📜results_final_shap.pickle
 ┃ ┃ ┣ 📂xgboost
 ┃ ┃ ┃ ┣ 📜results_final_complex.pickle
 ┃ ┃ ┃ ┣ 📜results_final_noint.pickle
 ┃ ┃ ┃ ┣ 📜results_final_shap.pickle
 ┃ ┃ ┃ ┣ 📜results_mp.pickle
 ┃ ┃ ┃ ┣ 📜results_noint.pickle
 ┃ ┃ ┃ ┗ 📜results_shap.pickle
 ┃ ┃ ┗ 📜y_dates.pickle
 ┃ ┗ 📂final_DK
 ┃ ┃ ┣ 📂baseline
 ┃ ┃ ┃ ┣ 📜results_ar1.pickle
 ┃ ┃ ┃ ┗ 📜results_ar_year_lag.pickle
 ┃ ┃ ┣ 📂elastic
 ┃ ┃ ┃ ┣ 📜results_final.pickle
 ┃ ┃ ┃ ┗ 📜results_mp.pickle
 ┃ ┃ ┣ 📂lasso
 ┃ ┃ ┣ 📂randomforest

 ┃ ┃ ┣ 📂ridge
 ┃ ┃ ┣ 📂weighted
 ┃ ┃ ┣ 📂xgboost
 ┣ 📜1_merge_robust.ipynb
 ┣ 📜2_descriptive_robust.ipynb
 ┣ 📜3_analysis_robust.ipynb
 ┣ 📜4_model_weights.ipynb
 ┣ 📜5_analysis_robust_conf.ipynb
 ┗ 📜6_results_robust.ipynb


