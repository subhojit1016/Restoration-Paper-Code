import pandas as pd
from scipy import stats
import numpy as np
confidence_level = 0.95


s=[0.0, 1.5, 3.0, 6.0]
e=[0.25, 0.5, 0.75, 0.90]
f1="40_SA"
f3="40"
f2="40R_4"
a="40"



for file_number in s:
    for file_number1 in e:
        file_name2 = f'{file_number}_Table_Network{f2}.txt_{file_number1}.csv'
        df2 = pd.read_csv(file_name2)  
        df2['P/SA'] = df2['P'] / df2['SA']
        df2['P Gap'] = df2['P'] / df2['SA'] - 1
        df2['NN Gap'] = df2['NN']/df2['SA'] - 1
        columns_to_confint = ['P Gap', 'NN Gap']
        cf=[]
        for column_name in columns_to_confint:
            data_mean = df2[column_name].mean()
            data_std = df2[column_name].std(ddof=1)  # Using ddof=1 for sample standard deviation
            sample_size = len(df2[column_name])
            
            standard_error = data_std / np.sqrt(sample_size)
            t_score = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)
            margin_of_error = t_score * standard_error
            confidence_interval = (int((round((data_mean - margin_of_error),2))*100), int((round((data_mean + margin_of_error),2))*100))
            cf.append(confidence_interval)
            
        columns_to_average = ['SA', 'P', 'NN', 'P Gap', 'NN Gap']
        average_array = df2[columns_to_average].mean().values
        average_dat = pd.DataFrame([average_array], columns=columns_to_average, index=None)
        average_dat.insert(0, 'p_value', file_number1)
        average_dat.insert(0, 's', file_number)
        average_dat['P Gap CI'] = [cf[0]]
        average_dat['NN Gap CI'] = [cf[1]]

        

        average_dat.to_csv(f'{file_number}_Stats_{a}_{file_number1}.csv', index=False)
        df2.to_csv(f'{file_number}_Data_{a}_{file_number1}.csv', index=False)
        

for file_number in s:
    for file_number1 in e:
        #file1 = f'{file_number}Computational_Table_Network{f1}.txt_{file_number1}.csv'
        file2 = f'{file_number}Computational_Table_Network{f2}.txt_{file_number1}.csv'
        #file3 = f'{file_number}Computational_Table_Network{f3}.txt_{file_number1}.csv'


        d2 = pd.read_csv(file2)

        columns_to_extract = ['States', 't1(sec)', 't2(sec)']
        df_extracted2 = d2[columns_to_extract].rename(columns={'States': 'States_'+ str(f2[:5]), 't1(sec)': 't1(sec)_' + str(f2[:5]), 't2(sec)': 't2(sec)_' + str(f2[:5])})

        merged = pd.concat([df_extracted2], axis =1)
        merged.insert(0, 'p_value', file_number1)
        merged.insert(0, 's', file_number)
        merged.to_csv(f'{file_number}_CT_{a}_{file_number1}.csv', index=False)



combined_df = pd.DataFrame()
for file_number in s:
    for file_number1 in e:
        file_name = f'{file_number}_CT_{a}_{file_number1}.csv'
        df = pd.read_csv(file_name, header=None, skiprows=1)  # Read the CSV file with one row
        combined_df = combined_df.append(df, ignore_index=True)


column = ['s','p_value','States_SA', 't1(sec)_SA', 't2(sec)_SA']

combined_df.columns = column
combined_df['SM Gap'] = combined_df['States_SM'] / combined_df['States_OM'] -1
combined_df['SA Gap'] = combined_df['States_SA']/combined_df['States_OM'] -1
combined_df['SM Gap t1'] = combined_df['t1(sec)_SM'] / combined_df['t1(sec)_OM'] - 1
combined_df['SA Gap t1'] = combined_df['t1(sec)_SA']/combined_df['t1(sec)_OM'] -1 
#combined_df['t2(sec)_OM/t2(sec)_SM'] = combined_df['t2(sec)_OM'] / combined_df['t2(sec)_SM']
#combined_df['t2(sec)_SA/t2(sec)_SM'] = combined_df['t2(sec)_SA']/combined_df['t2(sec)_SM']

combined_df = combined_df.round(2)
combined_df.to_csv(f'CT_{f1[:2]}.csv', index=False)


combined_df1 = pd.DataFrame()
for file_number in s:
    for file_number1 in e:
        file_name = f'{file_number}_Stats_{a}_{file_number1}.csv'
        df = pd.read_csv(file_name, header=None, skiprows=1)  # Read the CSV file with one row
        combined_df1 = combined_df1.append(df, ignore_index=True)

column1 = ['s','p_value','SA', 'P', 'NN', 'P Gap', 'NN Gap', 'P Gap CI', 'NN Gap CI' ]

combined_df1.columns = column1
combined_df1 = combined_df1.round(2)
combined_df1.to_csv(f'Stats_{f1[:2]}.csv', index=False)

