import pandas as pd

def label_messages(file_path):
    # Load the CSV file
    df = pd.read_excel(file_path)

    # Check if the necessary columns exist
    if 'w' not in df.columns or 'annotated_stance' not in df.columns:
        print("Error: CSV must contain 'w' and 'annotated_stance' columns.")
        return

    # Iterate through the messages
    for idx, row in df.iterrows():
        message = row['w']
        print(f"Message {idx + 1}: {message}")
        
        # Get the label from the user
        while True:
            label = input("Label this message (for/against/neutral): ").lower().strip()
            if label in ['for', 'against', 'neutral']:
                df.at[idx, 'annotated_stance'] = label
                break
            else:
                print("Invalid label. Please choose 'for', 'against', or 'neutral'.")

    # Save the updated dataframe to the CSV
    df.to_csv('annotated_df_dara', index=False)
    print("Labels have been saved to the CSV.")

# Example usage:
file_path = 'sample_messages.xlsx'
label_messages(file_path)
