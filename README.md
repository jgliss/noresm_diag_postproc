# Python tool for post analysis and visualisation of NorESM CAM diagnostics files

Written and tested in Python 3

## Requirements

In the following, all requirement packages are listed including version specifications, for which the tools were developed.

- pandas >= 0.23.0
- matplotlib >= 2.2.2
- seaborn >= 0.8.1
- numpy >= 1.14.0

__Optional__

- ipywidgets >= 7.2.1 (for interactive features -> see *widgets.py*)
- urllib (for direct downloads of ascii tables -> see notebook *download_tables.ipynb*)
- csv (for direct downloads of ascii tables -> see notebook *download_tables.ipynb*)

## Getting started

Please clone or download the repository. Please make sure that you have all requirements installed and open the jupyter notebook app.

```
$ jupyter notebook
```

This should open the notebook app in your default browser. From there, open one of the analysis notebook tools (currently it contains only one: ***analysis_tool.ipynb***) and follow the instructions in the introduction section.

### Preparation of tables 

You can download them manually or use notebook [download_tables.ipynb](https://github.com/jgliss/noresm_diag_postproc/blob/master/download_tables.ipynb) for creating local copies of result tables (based on a list of provided URL's)

### Additional source files

The tool(s) make use of additional functions and classes that can be found in the additional source files:

- ***helper_funcs.py***: functions for I/O, processing, analysis and visualisation of tables.
- ***widgets.py***: interactive features (currently not used)

## Repository structure

Python source files (*.py*) and notebooks (*.ipynb*) can be found in the top level directory. The repository contains two further directories:

### config (directory)

Contains additional configuration files. Currently

- ***var_groups.ini***: use this file to specify groups of variables, that you want to use in the analysis notebook

- ***var_info.csv***: use this file to assign more intuitive short names to variable IDs.

### example_data (directory)

Contains 6 example tables that are used in the post analysis notebooks for illustration purposes.
