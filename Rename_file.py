import pandas as pd
from scipy import stats
import numpy as np
confidence_level = 0.95


s=[0.0, 1.5, 3.0, 6.0]
e=[0.25, 0.5, 0.75, 0.90]
f1="16_SM"
#f3="I0_OM"
f2="16_SA"
a="16"



for file_number in s:
    for file_number1 in e:
        file_name1 = f'{file_number}_Table_Network{f1}.txt_{file_number1}.csv'
        file_name2 = f'{file_number}_Table_Network{f2}.txt_{file_number1}.csv'
        #file_name3 = f'{file_number}_Table_Network{f3}.txt_{file_number1}.csv'
        df1 = pd.read_csv(file_name1)  # Read the CSV file with one row
        df2 = pd.read_csv(file_name2)  
        #df3 = pd.read_csv(file_name3)  
        columns_to_add1 = ['Action_SA', 'SA']
        #columns_to_add2 = ['Action_OM', 'OM']
        df2_columns = df2[columns_to_add1]
        #df3_columns = df3[columns_to_add2]
        merged_df = pd.concat([df1, df2_columns], axis=1)
        #merged_df1 = pd.concat([merged_df, df2_columns], axis=1)
        #merged_df['P/SM'] = merged_df['P'] / merged_df['SM']
        #merged_df['NN/SM'] = merged_df1['NN']/merged_df1['SM']
        #merged_df1['P/SA'] = merged_df1['P'] / merged_df1['SA']
        #merged_df1['NN/SA'] = merged_df1['NN']/merged_df1['SA']
        merged_df['SA Gap'] = merged_df['SA'] / merged_df['SM'] - 1
        #merged_df1['SA Gap'] = merged_df1['SA']/merged_df1['OM'] - 1
        merged_df['P Gap'] = merged_df['P'] / merged_df['SM'] - 1
        merged_df['NN Gap'] = merged_df['NN']/merged_df['SM'] - 1
        columns_to_confint = ['SA Gap', 'P Gap', 'NN Gap']
        cf=[]
        for column_name in columns_to_confint:
            data_mean = merged_df[column_name].mean()
            data_std = merged_df[column_name].std(ddof=1)  # Using ddof=1 for sample standard deviation
            sample_size = len(merged_df[column_name])
            
            standard_error = data_std / np.sqrt(sample_size)
            t_score = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)
            margin_of_error = t_score * standard_error
            confidence_interval = (round((data_mean - margin_of_error),2), round((data_mean + margin_of_error),2))
            cf.append(confidence_interval)
            
        columns_to_average = [ 'SM', 'SA', 'P', 'NN', 'SA Gap', 'P Gap', 'NN Gap']
        average_array = merged_df[columns_to_average].mean().values
        average_dat = pd.DataFrame([average_array], columns=columns_to_average, index=None)
        average_dat.insert(0, 'p_value', file_number1)
        average_dat.insert(0, 's', file_number)
        #average_dat['SM Gap CI'] = [cf[0]]
        average_dat['SA Gap CI'] = [cf[0]]
        average_dat['P Gap CI'] = [cf[1]]
        average_dat['NN Gap CI'] = [cf[2]]

        

        average_dat.to_csv(f'{file_number}_Stats_{a}_{file_number1}.csv', index=False)
        merged_df.to_csv(f'{file_number}_Data_{a}_{file_number1}.csv', index=False)
        

for file_number in s:
    for file_number1 in e:
        file1 = f'{file_number}Computational_Table_Network{f1}.txt_{file_number1}.csv'
        file2 = f'{file_number}Computational_Table_Network{f2}.txt_{file_number1}.csv'
        #file3 = f'{file_number}Computational_Table_Network{f3}.txt_{file_number1}.csv'

        d1 = pd.read_csv(file1)
        d2 = pd.read_csv(file2)
        #d3 = pd.read_csv(file3)
        columns_to_extract = ['States', 't1(sec)', 't2(sec)']
        df_extracted1 = d1[columns_to_extract].rename(columns={'States': 'States_'+ str(f1[:5]), 't1(sec)': 't1(sec)_' + str(f1[:5]), 't2(sec)': 't2(sec)_' + str(f1[:5])})
        df_extracted2 = d2[columns_to_extract].rename(columns={'States': 'States_'+ str(f2[:5]), 't1(sec)': 't1(sec)_' + str(f2[:5]), 't2(sec)': 't2(sec)_' + str(f2[:5])})
        #df_extracted3 = d3[columns_to_extract].rename(columns={'States': 'States_'+ str(f3[:5]), 't1(sec)': 't1(sec)_' + str(f3[:5]), 't2(sec)': 't2(sec)_' + str(f3[:5])})
        merged = pd.concat([df_extracted1, df_extracted2], axis =1)
        merged.insert(0, 'p_value', file_number1)
        merged.insert(0, 's', file_number)
        merged.to_csv(f'{file_number}_CT_{a}_{file_number1}.csv', index=False)



combined_df = pd.DataFrame()
for file_number in s:
    for file_number1 in e:
        file_name = f'{file_number}_CT_{a}_{file_number1}.csv'
        df = pd.read_csv(file_name, header=None, skiprows=1)  # Read the CSV file with one row
        combined_df = combined_df.append(df, ignore_index=True)


column = ['s','p_value','States_SM', 't1(sec)_SM', 't2(sec)_SM', 'States_SA', 't1(sec)_SA', 't2(sec)_SA', 'States_OM', 't1(sec)_OM', 't2(sec)_OM']

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

column1 = ['s','p_value','SM', 'SA', 'P', 'NN', 'SA Gap', 'P Gap', 'NN Gap',  'SA Gap CI', 'P Gap CI', 'NN Gap CI' ]

combined_df1.columns = column1
combined_df1 = combined_df1.round(2)
combined_df1.to_csv(f'Stats_{f1[:2]}.csv', index=False)

