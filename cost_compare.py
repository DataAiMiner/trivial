import pandas as pd


actual_cost_df = pd.read_excel('./220307_valid_pred.xlsx', engine='openpyxl', index_col=False)
converted_accounts = []
for account in actual_cost_df['account_id']:
    converted_accounts.append(str(account).zfill(12))
actual_cost_df['account_id'] = converted_accounts

files_list = ['./till_03837.xlsx', './till_10524.xlsx', './till_33725.xlsx', './till_55825.xlsx', './till_75936.xlsx',
              './till_95447.xlsx', './till_99830.xlsx']
main_df_list = []
for file in files_list:
    df = pd.read_excel(file, engine='openpyxl', index_col=False)
    df_pred = df[df['예측 표시기'] == '예상']
    # print(df_pred.head(5))

    accounts = df_pred.columns[2:]
    # for account in ['000829023097']:
    for account in accounts:
        account_df_data = {}
        account_df_data['account_id'] = account
        account_df_data['usage_date'] = df_pred['usage_date'].reset_index(drop=True)
        account_df_data['actual_cost'] = actual_cost_df[actual_cost_df['account_id'] == account]['actual_cost'].reset_index(drop=True)
        account_df_data['predict_cost'] = df_pred[account].reset_index(drop=True)
        account_df = pd.DataFrame(account_df_data)
        main_df_list.append(account_df)

main_df = pd.concat(main_df_list)
print(main_df.head(5))
print(main_df.tail(5))
main_df.to_excel('./tableau_cost_pred_final.xlsx', sheet_name='전체결과', header=True, index=False)
