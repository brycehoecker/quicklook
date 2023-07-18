import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/ctauser/CameraSoftware/trunk/data_taking/')
from psct_toolkit import PsctTool


def plot_list(inl, outfile, show=False, scan_type="thresh"):
    fig = plt.figure() 
    plt.rc('axes', titlesize=40)
    plt.rc('axes',labelsize=30)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.rc('legend', fontsize=15)
    plt.rc('figure', titlesize=40)
    plt.figure(figsize=(12,9))
    colors = plt.cm.viridis(np.linspace(0.02,0.98,len(inl)))
   
    for i,run in enumerate(inl):
        inf = dataDir + str(run) + '_scan.txt'
        if(scan_type=="thresh"):
            names=["thresh","rate"]
        if(scan_type=="pe"):
            names=["pe","rate"]
        df = pd.read_csv(inf, header=None, sep=r"\s+", names = names)
        # adding a small floor to 0 rates so that we can see them on log plot
        df.loc[df.rate==0, 'rate'] = min(np.min(df.loc[df.rate>0, 'rate'])*0.1, 1e-2)
        plt.plot(df.loc[:,names[0]],df.rate, marker=".", ls="--", alpha=0.9, label='Scan '+str(run))
    
    if scan_type=="thresh":
        plt.gca().invert_xaxis()
    plt.yscale('log')
    plt.xlabel("Thresh value ({})".format(names[0]))
    plt.ylabel("Rate (Hz)") 
    #plt.title("Rate Scan")

    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.01,1), loc='upper left')
    plt.tight_layout()
    if show:
        plt.show()
    
    plt.savefig('{}/{}'.format(outputDir,outfile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot rate scan data')
    parser.add_argument('-l', '--list', default=None, nargs='+', help='Assume all files are in /data/local_outputDir as XXXXXX_scan.txt, you can provide a list of XXXXXX run numbers')
    parser.add_argument('-st', '--scan_type', default="thresh", help="Scan type (thresh or p.e.)")
    parser.add_argument('-o', '--output', help='Name of the output image file. Default is RateScan_listOfRuns.png')
    parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")
    args = parser.parse_args()
    
    outputDir = '/data/analysis_output/rateScanPlots'
    dataDir = '/data/local_outputDir/'
    
    # Setting the outfile
    if args.output is None:
        outfile = 'RateScan'
        for num in args.list:
            outfile = outfile + '_' + str(num)
        outfile = outfile + '.png'
    else:
        outfile = args.output

    show = args.interactive
    scan_type = args.scan_type
    if args.list is not None:
        plot_list(args.list, outfile, show=show, scan_type=scan_type)