#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Development script for direct download of result files
"""
import urllib
import csv
import os

url = "http://ns2345k.web.sigma2.no/noresm_diagnostics/NF1850_CTRL_f19_f19_r265_20180412/CAM_DIAG/yrs5to30-obs/set1/table_GLBL_ANN_obs.asc"

def download_cams_diag_result_csv(url, save_dir):
    
    response = urllib.request.urlopen(url) # it's a file like object and works just like a file
    lines = [l.decode("UTF-8").strip() for l in response.readlines()]
    
    s =lines[2]
    if not s.startswith("TEST"):
        raise IOError("URL is likely not a valid diagnostics file since it "
                      "does not include TEST CASE specification in 3 line")
    filename = s.split("TEST CASE:")[-1].strip() + '.csv'
    outfile = os.path.join(save_dir, filename)
    
    if os.path.exists(outfile):
        raise IOError("CSV file {} already exists in directory".format(outfile))
    with open(outfile, "w") as output:
        writer = csv.writer(output)
        for val in lines:
            writer.writerow([val])
    return lines
    
lines = download_cams_diag_result_csv(url, ".")
     

    
    
