"""
MASTER THESIS PYTHON CODE
by Luis Barrantes Coloma
Universitat Politècnica de Catalunya
Escola Tècnica Superior d'Enginyeria Industrial de Barcelona
16 / 06 / 2021
"""
#%%
#The modules used on the program
import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%%
def set_comparisons(lsets,sun_inst = 1, wind1_inst = 1, wind2_inst = 1, write = -1):
    """
    Main program for the first practical section:
        
    Given a list of sets, create the different set comparisons between them using tham as sun + wind and wind only (both cases);
    one set will not be compared with itself.
    """
    fw = open('Set comparisons.csv','w')    #File where we store the comparisons done
    dic_sets = {}                           #We also save them as a variable

    fw.write('Sun + wind set;Wind only set;Demand coverage\n')
    for i in range(len(lsets)):
        for j in range(len(lsets)):
            if i!=j:
                #We convert the downloaded sets to be able to work with them; creating a temporary file ending in _ready.csv           
                set_i = df_structure(lsets[i]+'.csv')
                set_i_short = set_i[:-4]
                set_i = set_i[:-4]+'_ready.csv'
                set_j = df_structure(lsets[j]+'.csv')
                set_j_short = set_j[:-4]
                set_j = set_j[:-4]+'_ready.csv'

                #For each combination of sets we merge them into one set and do the different studies
                df = df_merge(set_i, set_j)
                filename = 'Comp-'+set_i_short +'-'+ set_j_short+'.csv'
                df2, dmed = df_edit (df.copy(),sun_inst, wind1_inst, wind2_inst, filename, write)
                dic_sets[filename[:-4]] = df2

                #We write in another file the overall demand coverage:
                fw.write(set_i_short+';'+set_j_short+';'+str(dmed)+'\n')
    fw.close()
    return dic_sets

#%%
def df_structure(set_name):
    """
    We transform the downloaded file into one we can work with.
    """
    #If the file we are going to create it already exists, we just use it
    if os.path.isfile(set_name[:-4]+'_ready.csv'):  
        return set_name
    
    fr_set = open (set_name,'r')
    fr_demand = open ('Energy demand 2007-2016.csv','r')

    fw = open (set_name[:-4]+'_ready.csv','w')

    #We store the location data to know the exact parameters used in obtaining it; even though we will not be using it in the project
    dparam = {}
    dparam['latitude'] = fr_set.readline().replace('"','').split()[3]
    dparam['longitude'] = fr_set.readline().replace('"','').split()[3]
    dparam['elevation'] = fr_set.readline().replace('"','').split()[2]
    fr_set.readline()
    fr_set.readline()
    fr_set.readline()
    dparam['slope'] = fr_set.readline().replace('"','').split()[1]
    dparam['azimuth'] = fr_set.readline().replace('"','').split()[1]
    dparam['nom_power'] = fr_set.readline().replace('"','').split()[8]
    dparam['system_losses'] = fr_set.readline().replace('"','').split()[3]

    fw.write('lat (º);'+ dparam['latitude'] + ';long (º);' + dparam['longitude'] + ';elev (m);' + dparam['elevation'] + '\n')
    fw.write('slope (º);'+ dparam['slope'] + ';azimuth (º);' + dparam['azimuth'] + ';nom power (kW);' + dparam['nom_power'] + ';syst losses (%);' + dparam['system_losses'] +'\n\n')
    
    fw.write('time;P;WS10;WS120;Demand\n')
    fr_demand.readline() 
    fr_set.readline()
    
    #We start merging all the 10 years data (sun + wind + demand)
    while True:
        l_set = fr_set.readline().strip('\n').split(',')
        l_dem = fr_demand.readline().strip('\n').split(';')

        if len(l_set) == 1: #After all data we have a blank space, then we exit the loop
            break

        dem = l_dem[1]

        #We calculate the wind speed at 120 meters; more information in the memory's section: 5.2. Preprocessing of the project
        v120val = float(l_set[5]) * (np.log(120/0.1) / np.log(10/0.1))
        v120 = str(v120val)[:4]

        #We remove unnecessary values and add the demand to the individual set
        fw.write(l_dem[0]+';'+l_set[1]+';'+l_set[5]+';'+v120+';'+dem+'\n')

    fr_set.close()
    fr_demand.close()
    fw.close()
    return set_name

#%%
def df_merge(solar_set, wind_set):
    """
    We create the combination of sets (sun + wind and wind only),  more information in the memory's section: 5.2. Preprocessing section.
    """
    set_name = solar_set.rstrip('ready.csv').strip('_')+'/'+wind_set.rstrip('ready.csv').strip('_')+':'
    df_sol = pd.read_csv(solar_set, sep = ';',skiprows = 3)
    df_wind = pd.read_csv(wind_set, sep = ';',skiprows = 3)

    del df_sol['WS10']
    df_sol.columns = [set_name+'time','Sun_power','Wind1_speed','Demand']
    df_wind = df_wind['WS120']
    df_sol.insert(3,'Wind2_speed',df_wind) 
    return df_sol

#%%
def df_edit(df, sun_inst, wind1_inst, wind2_inst = 0, filename = -1, write = -1):
    """
    Uses the data obtained from df_merge and does the ponderation study; more information in the memory's section: 5.2. Preprocessing.
    """
    #If we only have 1 value of installed wind we set the two at the same value
    if wind2_inst == 0: 
        wind2_inst = wind1_inst

    #Total installed sun power; we have it in W gen / kW installed; we convert it to MW gen / MW installed
    df['Sun_power'] = df['Sun_power'] / 1000 * sun_inst

    #Wind power, we calculate it using the power curve as mentioned in the memory's section: 5.2. Preprocessing
    df.insert(3,'Wind1_power',wind_calc(df['Wind1_speed'],wind1_inst))
    df.insert(5,'Wind2_power',wind_calc(df['Wind2_speed'],wind2_inst))    

    #We create the ponderation columns, firts we need to obtain the maximum values of all generations and demand
    solar_max_power = df.loc[:]['Sun_power'].max()
    wind1_max_power = df.loc[:]['Wind1_power'].max()
    wind2_max_power = df.loc[:]['Wind2_power'].max()
    max_demand = df.loc[:]['Demand'].max()
    
    #We insert the ponderation columns; the generations scaled between 0 and 1 and the demand between 0 and 3; for more information consult the memory's section: 5.2. Preprocessing
    df.insert(2,'Sun_pond',df['Sun_power']/solar_max_power)
    df.insert(5,'Wind1_pond',df['Wind1_power']/wind1_max_power)
    df.insert(8,'Wind2_pond',df['Wind2_power']/wind2_max_power)
    df['Demand_pond'] = df['Demand']/max_demand*3
    
    #We proceed to calculate the total generation ponderation and demand coverage ponderation
    df['Gen_pond'] = df['Sun_pond'] + df['Wind1_pond'] + df['Wind2_pond']
    df['Dem_cov'] =  df['Gen_pond'] / df['Demand_pond']

    #Finally, we calculate the overall demand coverage
    Dem_cov_med = np.mean(df['Dem_cov']) 
  
    #Write the dataframe in a .csv format, with a default name if not given any
    if write == 1:
        if not isinstance(filename, str):
            filename = 'set.csv'
        df.to_csv(filename,sep=';')

    return df, Dem_cov_med

#%%
def wind_calc(column,num_gen):
    """
    Auxiliary function of df_edit; calculates the wind power from the wind speed using the equation described in the memory's section: 5.2. Preprocessing
    """
    l = []
    for v in column:
        if v < 3.5 or v > 25:
            l.append(0*num_gen)
        elif v <= 14:
            a = 0.0952 * v - 0.3333 
            l.append(a*num_gen)
        else: 
            l.append(1*num_gen)
    return l

#%%
def graph(df):
    """
    This function plots the graphs used in the hourly ponderation study
    """
    #We temporarily rename the columns for easier use
    df = df.rename(columns = {'Dem_cov': 'pond'})
    df = df.rename(columns = {df.columns[0]: 'time'})

    X = np.linspace(0, 23, 24, endpoint=True)
    l_average = [] 
    l_average_sun = []
    l_average_wind1 = []
    l_average_wind2 = []
    l_dem = []
    
    #We calculate the average values for each hour of the 10 years
    for e in range(24):
        e = str(e)
        if len(e) == 1:
            e = '0'+ e
        l_average.append(df[df['time'].str[-2:]==e]['pond'].mean())
        l_average_sun.append((df[df['time'].str[-2:]==e]['Sun_pond']).mean())
        l_average_wind1.append((df[df['time'].str[-2:]==e]['Wind1_pond']).mean())
        l_average_wind2.append((df[df['time'].str[-2:]==e]['Wind2_pond']).mean())
        l_dem.append(df[df['time'].str[-2:]==e]['Demand_pond'].mean())
    
    arr_average = np.array(l_average)
    
    #Additive generation source vs demand plot
    plt.figure()
    plt.plot(X, np.array(l_average_sun) , color = 'orange', linewidth = 1, linestyle = '-', label = 'Sun production')
    next_layer = np.array(l_average_sun) + np.array(l_average_wind1)
    plt.plot(X, next_layer , color = 'darkblue', linewidth = 1, linestyle = '-', label = 'Sun + Wind 1 production')
    next_layer +=  np.array(l_average_wind2)
    plt.plot(X, next_layer , color = 'cyan', linewidth = 1, linestyle = '-', label = 'Sun + Wind 1 + Wind 2 production')
    plt.plot(X, l_dem , color = 'red', linewidth = 1, linestyle = '-', label = 'Power demand')
    plt.xlim(0,23)
    plt.xticks(list(range(24)))
    plt.title('Additive generation source vs demand')
    plt.legend(loc = 'upper left')
    plt.show()
    
    #Percenage dof demand covered plot
    plt.figure()
    plt.plot(X, arr_average , color = 'green', linewidth = 1, linestyle = '-', label = 'Percentage of demand covered')
    plt.xlim(0,23)
    plt.xticks(list(range(24)))
    plt.ylim(0,1)
    plt.title('Percentage of demand covered')
    plt.legend(loc = 'best')
    plt.show()
    
#%%
def storage(d_frame,threshold,hours = 24):
    """
    The main code for the second practical section (the storage study); we use the same sets created in the set_comparisons part
    """
    #We remove unnecessary columns
    df = d_frame.rename(columns = {d_frame.columns[0]: 'time'})
    df.pop('Sun_pond')
    df.pop('Wind1_speed')
    df.pop('Wind1_pond')
    df.pop('Wind2_speed')
    df.pop('Wind2_pond')
    df.pop('Demand_pond')
    df.pop('Gen_pond')
    df.pop('Dem_cov')
    
    #We calculate the installed power coefficients for each time period; for more information chech the memory's section: 7.3. Storage values and coefficients calculation. 
    max_thresh_sun = 1000000
    max_thresh_wind = 1000000
    b_sun = (0, max_thresh_sun) 
    b_wind = (0,max_thresh_wind)
    bounds = (b_sun,b_wind,b_wind)
    con = {'type':'eq', 'fun': constraint}
    coef_0 = [10,10,10]   
    daily_storage = []
    
    for d in range (int(df.shape[0]/(hours))):
        global df_slice
        df_slice =  df.iloc[d * hours: (d + 1) * hours]
        if df_slice.shape[0] != hours:
            break
        params = minimize(objective, coef_0, bounds = bounds, constraints = con)
        daily_storage.append([d,params.fun,params.x])

    
    df_storage = pd.DataFrame.from_records(daily_storage)
    coef_sun = 0 
    coef_w1 = 0 
    coef_w2 = 0 
    
    for e in df_storage[2]:
        coef_sun += e[0]
        coef_w1 += e[1]
        coef_w2 += e[2]
    coef_sun /= df_storage.shape[0]
    coef_w1 /= df_storage.shape[0] 
    coef_w2 /= df_storage.shape[0]
    
    chosen_coefs = [coef_sun,coef_w1,coef_w2]

    #With the obtained values we proceed to calculate the new storage values; where we will purposely leave a certain % out as a voluntary loss
    new_sto = []
    for d in range (int(df.shape[0]/(hours))):
        df_slice =  df.iloc[d * hours: (d + 1) * hours]
        if df_slice.shape[0] != hours:
            break  
        new_sto.append(objective(chosen_coefs))
    new_sto = np.array(new_sto)
    max_storage = np.percentile(new_sto,threshold*100)
    values = chosen_coefs

    #Maximum storage each period of time plot (full plot)   
    plt.figure()
    N, bins, patches = plt.hist(new_sto, bins = [x*new_sto.max()/100 for x in range(101)], edgecolor = 'k')
    plt.title('Maximum storage each period of time (through regression)')
    plt.ylabel('Number of periods')
    plt.xlabel('Storage value (MW)')
    
    for i in range (len(patches)):
        if bins[i] >=  max_storage:
            patches[i].set_facecolor('k')
        else:
            patches[i].set_facecolor('b')
    plt.show()
    
    #Maximum storage each period of time plot (after capping)
    plt.figure()
    plt.hist(new_sto, color='blue', bins = [x*max_storage/100 for x in range(101)], edgecolor = 'k')
    plt.title('Maximum storage each period of time (through regression)')
    plt.ylabel('Number of periods')
    plt.xlabel('Storage value (MW)')
    plt.show()
    
    #Here we take the maximum storage value keeping in mind the threshold and the maximum storage value
    df['New_generation(MW)'] = df['Sun_power'] * values[0] + df['Wind1_power'] * values[1] + df['Wind2_power'] * values[2]
    df['Hourly_deficit(MW)'] = (df['New_generation(MW)'] - df['Demand']) #Positive => Excess ; Negative => Defect
    df = storage_variation(df,max_storage)
    df['Storage_excess(MW)'] = df['Hourly_deficit(MW)']*(df['Storage_value(MW)']==max_storage)
    df['Storage_deficit(MW)'] = abs(df['Hourly_deficit(MW)']*(df['Storage_value(MW)']==0))
    
    #We plot the hourly storage study's 3 plots deficit, storage used and excess
    max_ex = int((df['Storage_excess(MW)'].max()//10000)+1) *10000
    max_def = int((df['Storage_deficit(MW)'].max()//5000)+1) * 5000
    df_aux = df[df['Storage_value(MW)'] < 0.99* df['Storage_value(MW)'].max()] #We crop tha 0 and maximum values as we consider them deficit or excess respectively
    df_aux2 = df_aux[df_aux['Storage_value(MW)'] > 0.1]
    
    plt.figure()    
    plt.hist(df_aux2['Storage_value(MW)'], color='blue', bins = [x*max_storage/50 for x in range(51)], edgecolor = 'darkblue')
    plt.title('Hourly maximum storage used (MW)')
    plt.ylabel('Number of hours')
    plt.xlabel('Storage value (MW)')
    plt.show()

    plt.figure()
    plt.hist(df[df['Storage_excess(MW)']>0.1]['Storage_excess(MW)'], color='green', bins = [x*max_ex/50 for x in range(51)], edgecolor = 'darkgreen')
    plt.title('Hourly maximum excess generated (MW)')
    plt.ylabel('Number of hours')
    plt.xlabel('Excess value (MW)')
    plt.show()

    plt.figure()
    plt.hist(df[df['Storage_deficit(MW)']>0.1]['Storage_deficit(MW)'], color='red', bins = [x*max_def/50 for x in range(51)], edgecolor = 'darkred')
    plt.title('Hourly maximum deficit (MW)')
    plt.ylabel('Number of hours')
    plt.xlabel('Deficit value (MW)')
    plt.show()
    
    #We calculate the excess and deficit streaks
    l_deficit = []
    l_excess = []
    streak_start = ''
    streak = 0
    previous_day = 'N' #N = neutral; E=excess; D=Deficit
    
    for hour in range (int(df.shape[0])):
        if df['Storage_excess(MW)'][hour] != 0 : #If today is a Excess period
            if previous_day == 'E': #E streak continues
                streak += 1
            
            elif previous_day == 'D': #D streak ends, E streak starts in this order
                l_deficit.append([streak_start, streak]) 
                streak_start = df['time'][hour]
                streak = 1                
            
            else: #E streak starts
                streak_start = df['time'][hour]
                streak = 1
                
            previous_day = 'E'  
            
        elif df['Storage_deficit(MW)'][hour] != 0: #If today is a Deficit period
            #print(df['time'][hour] +' is D')
            if previous_day == 'D': #D streak continues
                streak += 1
            
            elif previous_day == 'E': #E streak ends, D streak starts in this order
                l_excess.append([streak_start, streak]) 
                streak_start = df['time'][hour]
                streak = 1                
            
            else: #D streak starts
                streak_start = df['time'][hour]
                streak = 1
                
            previous_day = 'D'  
                     
        else:  #If today is a Neutral period
            if previous_day == 'D': #D streak ends
                l_deficit.append([streak_start, streak]) 
            
            elif previous_day == 'E': #E streak ends
                l_excess.append([streak_start, streak]) 
           
            # If previous day was Neutral we do nothing
            previous_day = 'N'      
        
    if l_deficit == []: #In the case there is no deficit
        l_deficit.append(['NONE',0])
        
    if l_excess == []:  #In the case there is no excess
        l_excess.append(['NONE',0])   
    
    #We plot the excess/deficit stereaks
    dfe = pd.DataFrame.from_records(l_excess)
    dfd = pd.DataFrame.from_records(l_deficit)    
    max_x = ((max(dfe[1].max(),dfd[1].max())//10)+1) #We get the x ticks over the latest multiple of five of the maximum
    
    plt.figure()
    bins = [x*5 for x in range(max_x)]
    plt.hist(dfe[1], color='green', bins = bins, edgecolor = 'darkgreen')
    plt.title('Excess streaks (hours)')
    plt.ylabel('Number of streaks')
    plt.xlabel('Streak length')
    plt.show()


    plt.figure()
    bins = [x*5 for x in range(max_x)]
    plt.hist(dfd[1], color='red', bins = bins, edgecolor = 'darkred')
    plt.title('Deficit streaks (hours)')  
    plt.ylabel('Number of streaks')
    plt.xlabel('Streak length')
    plt.show()

    return (df_storage, df, new_sto ,max_storage, values, dfe, dfd)
       
#%%
"""
This three functions are the ones used in the minimization calculation; in orther they are the objective function, the constraint used 
and a function to do the accumulative sum in each row
"""            

def objective (c):
    """
    The function to minimize (the maximum storage)
    """
    global df_slice
    gen = df_slice['Sun_power'] * c[0] + df_slice['Wind1_power'] * c[1] + df_slice['Wind2_power'] * c[2]
    deficit = gen - df_slice['Demand'] 
    cum_deficit = abs(deficit.cumsum())
    return cum_deficit.max()                                                                                                                                                                             

def constraint(c):
    
    """
    The constraint to make the daily avg generation equal to the daily avg demand
    """
    global df_slice
    return df_slice['Sun_power'].sum() * c[0]  +df_slice['Wind1_power'].sum() * c[1] + df_slice['Wind2_power'].sum() * c[2] - df_slice['Demand'].sum()  
    
def storage_variation(df,storage):
    arr = []
    for i in range(df.shape[0]):
        if i == 0:
            arr.append( min(max(df['Hourly_deficit(MW)'][i],0),storage))
        else:
            new_storage = min(max(df['Hourly_deficit(MW)'][i]+arr[-1],0),storage)
            arr.append(new_storage)
    df['Storage_value(MW)'] = arr
    return df    
    
#%%
"""
One example for a set study
"""
#We create all the comparisons of these sets, and create the .csv files
lsets = ['set_Barcelona','set_Madrid','set_Sahara','set_Scotland','set_Coruña']
print('Creating Sets...')
dic_sets = set_comparisons(lsets,sun_inst = 1, wind1_inst = 1, wind2_inst = 1, write = -1) #This also does the ponderation calculations

#We choose this set to study (Barcelona-Madrid)
lstud = ['Comp-' + lsets[0] + '-' + lsets[1]]
print(dic_sets[lstud[0]])

#We create the plots used in the ponderation study
print('Creating Graphs...')
graph(dic_sets[lstud[0]])

# #We do the storage analysis
print('Storage analysis...')
df1, df2, new_sto,sto, values,dfe, dfd = storage(dic_sets[lstud[0]], 0.75, hours = 7*24)
print('Storage: %f\nCoefficients: '%sto,values)




