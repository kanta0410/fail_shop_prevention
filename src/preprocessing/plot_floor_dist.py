import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    df = pd.read_csv('data/processed/nagoya.csv')
    
    # 階数の外れ値を除外（-2階から10階まで）して見やすくする
    df_plot = df[(df['floor_num'] >= -2) & (df['floor_num'] <= 10)].copy()
    
    # 階数を整数に変換して見やすく
    df_plot['floor_num'] = df_plot['floor_num'].astype(int)
    
    # プロットの設定
    plt.figure(figsize=(12, 6))
    
    # sns.countplotで階層ごとの件数を is_closed で色分け
    ax = sns.countplot(
        data=df_plot, 
        x='floor_num', 
        hue='is_closed', 
        palette={0: '#1f77b4', 1: '#d62728'}  # 青=存続, 赤=廃業
    )
    
    plt.title('Distribution of Stores by Floor and Status (-2F to 10F)')
    plt.xlabel('Floor Number')
    plt.ylabel('Count')
    
    # 凡例の設定
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Alive (0)', 'Closed (1)'], title='Status')
    
    # 1階が圧倒的に多いため、Y軸を対数スケールにして他の階層も見やすくする設定（オプションとして追加）
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    
    # 保存
    output_dir = 'data/raw/nagoya_eda_3'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'floor_distribution_by_status.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f'Distribution plot successfully saved to {output_path}')

if __name__ == "__main__":
    main()
