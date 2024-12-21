import pandas as pd

def process_csv(file_name):
    data = pd.read_csv(file_name)
    last_100 = data.iloc[-100:]
    # Calculate the average of the second column (index 1)
    average_value = last_100.iloc[:, 1].mean()
    
    # Create a dictionary to hold the results with the filename
    result = {'File': file_name, 'Average_last_100': average_value}
    
    return result

# List of CSV files
#csv_files = ['Finally N=4.csv', 'Finally N=8.csv', 'Finally N=12.csv', 'Finally N=16.csv',"Finally N=20.csv", 'Finally N=24.csv', 'Finally N=28.csv', 'Finally N=30.csv', 'Finally N=40.csv'] 
#csv_files=['N=4.csv','N=8.csv',"N=12.csv","N=16.csv","N=20.csv","N=24.csv","N=28.csv","N=32.csv"]
# Process each file and store the results
csv_files=['N10.csv']
results = []
for file in csv_files:
    result = process_csv(file)
    results.append(result)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv = 'Average_values_from_last_100.csv'
results_df.to_csv(output_csv, index=False)

print(f'Results have been saved to {output_csv}')
