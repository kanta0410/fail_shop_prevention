import pandas as pd
import glob

def main():
    for f in glob.glob('data/**/*.csv', recursive=True):
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'is_closed' in df.columns:
                print(f"{f}: total={len(df)}, closed={df['is_closed'].sum()}")
            elif 'business_status' in df.columns:
                closed_cnt = df['business_status'].isin(['CLOSED_PERMANENTLY', 'NOT_FOUND_CLOSED']).sum()
                print(f"{f}: total={len(df)}, closed_by_status={closed_cnt}")
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
