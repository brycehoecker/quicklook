import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse
import datetime
import os

# Plots temperature over time for each module in a run or thresh scan
# Excludes modules with temperatures below 4C or above 100C
# Run with -i

# Create a list of module numbers present in the temperature file       
def list_modules(data):
    module_numbers = set()
    for sensor in data.keys():
        if sensor == 'timestamp_start':
            continue
        if sensor == 'timestamp_end':
            continue
        module_numbers.add(sensor.split('_')[0])
    return module_numbers

# Create a dictionary of key:Module value:avg_temperatures 
def avg_temps(data,module_numbers):
    avg_temp_per_module = {}
    for mod in module_numbers:
        individual_sensor_values = []
        for i in range(4):
            values = data[str(mod)+'_ADC'+str(i)]
            if all(val<100 for val in values):
                if all(val>4 for val in values):
                    individual_sensor_values.append(values)
        average_sensor_values = [sum(col) / float(len(col)) for col in zip(*individual_sensor_values)]
        if average_sensor_values != []:
            avg_temp_per_module[mod] = average_sensor_values
    print(avg_temp_per_module)
    return avg_temp_per_module

# Plot average temperatures vs time
def make_plot(temps,date,times,ID,type):
    mod_location = [[4,5,1,3,2],[103,125,126,106,9],[119,108,110,121,8],[115,123,124,112,7],[100,111,114,107,6]]
    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = [cmap(0),cmap(0.25),cmap(0.5),cmap(0.75),cmap(0.99)]
    #colors = ['red','blue','green','purple','yellow']
    #linestyles = ['-',(0,(7,2,1,2)),(0,(5,2,1,2)),(0,(3,2,1,2)),':']
    markers = ['o', 's', 'v', 'd', 'X']
    plt.figure(figsize=(40,30))
    fig = plt.figure()
    for mod in sorted(temps.keys()):
        for row in range(5):
            if int(mod) in mod_location[row]:
                marker = markers[row]
                #color = colors[row]
                for col in range(5):
                    if int(mod) ==  int(mod_location[row][col]):
                        color = colors[col]
                        #marker = markers[col]
                        #linestyle = linestyles[col]
        #if mod in ['2','9','8','7','6']:
        #    color = 'purple'
        #elif mod in ['3','106','121','112','107']:
        #    color = 'blue'
        #elif mod in ['1','126','110','124','114']:
        #    color = 'green'
        #elif mod in ['5','125','108','123','111']:
        #    color = 'yellow'
        #elif mod in ['4','103','119','115','100']:
        #    color = 'red'
        #else:
        #    color = 'black'
        plt.plot(times, temps[mod], label='Module ' + str(mod), color=color, marker=marker)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time')
    plt.ylabel('Temperature (C)')
    if type:
        plt.title('Temperature vs Time: Run' + str(ID) + '\nDate ' + str(date))
    else:
         plt.title('Temperature vs Time: Scan' + str(ID) + '\nDate ' + str(date))
    plt.legend(bbox_to_anchor=(1.01,1), loc='upper left')  
    plt.tight_layout()
    plt.savefig('{}/{}'.format(outputDir,outfile))    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="run number", help="Run number for a data run or thresh scan")
    args = parser.parse_args()
    ID = args.i

    dataDir = '/data/local_outputDir/'
    outputDir = '/data/analysis_output/temp_plots'
    outfile = '{}_tempVStime.pdf'.format(ID)

    filename = dataDir + str(ID) + '_temperatures.txt'
    data = pd.read_csv(filename)
    shape = data.shape
    #times = [data['timestamp_start'][i][11:] for i in range(shape[0])]
    times = [data['timestamp_start'][i] for i in range(shape[0])]
    times = mdates.num2date(mdates.datestr2num(times))
    date = datetime.date.fromisoformat(data['timestamp_start'][1][:10])

    modules = list_modules(data)
    temps = avg_temps(data,modules)

    run = os.path.exists(dataDir + 'run' + str(ID) + '.fits')
    scan = os.path.exists(dataDir + str(ID) + '_scan.txt')
    
    if run:
        make_plot(temps, date, times, ID, True)
    elif scan:
        make_plot(temps, date, times, ID, False)
    else:
        print('Something is wrong')
        sys.exit(ex)

