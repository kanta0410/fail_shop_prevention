import pandas as pd
import re
import yaml
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def categorize_store(name):
    name = str(name).lower()
    
    # Fast Food
    if re.search(r'マクドナルド|mcdonald|モスバーガー|mos burger|ケンタッキー|kfc|吉野家|yoshinoya|すき家|sukiya|松屋|matsuya|なか卯|nakau|ミスタードーナツ|mister donut|coco壱|coco ichibanya|バーガーキング|burger king|サブウェイ|subway|ほっともっと|hotto motto|aoki\'s pizza|アオキーズ|pizza-la|ピザーラ|domino|ドミノ|baskin robbins|サーティワン', name):
        return 'Fast Food'
    
    # Ramen
    if re.search(r'ラーメン|麺|ramen|noodle|つけ麺|担々麺|支那そば|ふくろう|一蘭|来来亭|rai rai tei|ラーメン福|sugakiya|寿がきや|スガキヤ|丸源|魁力屋|一刻魁堂|藤一番', name):
        return 'Ramen'
    
    # Sushi
    if re.search(r'寿司|鮨|sushi|はま寿司|hamazushi|くら寿司|kura sushi|スシロー|sushiro|丸忠|maruchu|魚魚丸|totomaru|銀のさら', name):
        return 'Sushi'
    
    # Cafe
    if re.search(r'カフェ|喫茶|coffee|珈琲|tea|ティー|cafe|ドトール|doutor|スターバックス|starbucks|コメダ|komeda|タリーズ|tully|星乃珈琲|支留比亜|おかげ庵|サンマルク|st marc', name):
        return 'Cafe'
    
    # Izakaya
    if re.search(r'居酒屋|酒場|串|焼鳥|焼鳥|バル|旨|呑|旬|割烹|炉ばた|izakaya|yakitori|beer|ビール|大吉|鳥貴族|torikizoku|風来坊|山ちゃん|木村屋|白木屋|魚民|笑笑|はなの舞|つぼ八', name):
        return 'Izakaya'
    
    # Yakiniku
    if re.search(r'焼肉|yakiniku|ホルモン|hormone|カルビ|kalbi|あみやき亭|amiyakitei|牛角|gyukaku|肉と米|スギモト|木こり家', name):
        return 'Yakiniku'
    
    # Western
    if re.search(r'ステーキ|steak|ハンバーグ|hamburg|パスタ|pasta|イタリアン|italian|フレンチ|french|洋食|ビストロ|bistro|ジョリーパスタ|jolly pasta|サイゼリヤ|saizeriya|ガスト|gusto|デニーズ|denny|ブロンコビリー|bronco billy|ロイヤルホスト|royal host|びっくりドンキー|bikkuri donkey|マリノ|marino|pastore|パステル', name):
        return 'Western'
    
    # Japanese
    if re.search(r'和食|うどん|udon|そば|soba|蕎麦|天ぷら|tempura|とんかつ|tonkatsu|かつ|katsu|定食|弁当|benton|丼|donburi|懐石|kaiseki|しゃぶしゃぶ|shabu|木曽路|kisoji|サガミ|sagami|どんどん庵|dondonan|街かど屋|machikadoya|まるは|なかむら|美濃路|みのじ|丸亀製麺|はなまる', name):
        return 'Japanese'
    
    # Chinese
    if re.search(r'中華|chinese|餃子|gyoza|飯店|飯店|閣|園|楼|軒|王将|ohsho|バーミヤン|bamiyan|味仙|misen|上海', name):
        return 'Chinese'
    
    # Sweets/Bakery
    if re.search(r'スイーツ|sweets|ケーキ|cake|菓子|パン|bakery|デザート|dessert|アイス|ice|ジェラート|チョコ|チョコレート|chocolate|ドーナツ|donut|クレープ|crepe|シャトレーゼ|chateraise', name):
        return 'Sweets/Bakery'
    
    # Asian
    if re.search(r'ベトナム|vietnam|インド|india|タイ|thai|アジアン|asian|ダウラギリ|ネパール|nepal|ケバブ|kebab|韓国|korea', name):
        return 'Asian'
    
    # Bar/Pub/Snack
    if re.search(r'bar|パブ|pub|スナック|snack|酒房|ラウンジ|lounge', name):
        return 'Bar/Pub/Snack'

    return 'Other'

def main():
    # Load config for paths
    with open('config/config.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_path = os.path.join(config['data_paths']['processed'], 'nagoya_analysis_warehouse.csv')
    output_path = os.path.join(config['data_paths']['processed'], 'nagoya_analysis_warehouse_categorized.csv')
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Categorizing stores...")
    df['category'] = df['name'].apply(categorize_store)
    
    # Print statistics
    stats = df.groupby('category')['is_closed'].agg(['count', 'mean']).sort_values('count', ascending=False)
    stats.columns = ['store_count', 'closure_rate']
    print("\nCategory Statistics:")
    print(stats)
    # 2. 均等な「層化分割カテゴリ」の作成 (5グループ)
    # 「廃業率」と「業態（business_type）」の両方が均等になるように分割
    # 層化のキーとして「廃業フラグ_業態」の組み合わせを作成
    df['stratify_key'] = df['is_closed'].astype(str) + "_" + df['category']
    
    # サンプル数が極端に少ない組み合わせ（5未満）は、StratifiedKFoldでエラーにならないよう調整
    counts = df['stratify_key'].value_counts()
    df['stratify_key'] = df['stratify_key'].apply(lambda x: x if counts[x] >= 5 else 'rare')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['balanced_group'] = 0
    
    for i, (_, val_idx) in enumerate(skf.split(df, df['stratify_key'])):
        df.loc[val_idx, 'balanced_group'] = i + 1
    
    # 3. 業態のラベルエンコーディング (Label Encoding)
    # ワンホットによる次元爆発を防ぐため、数値を割り振る
    le = LabelEncoder()
    df['category_label'] = le.fit_transform(df['category'])
    
    # ラベルの対応関係を表示
    print("\n--- Category Label Mapping ---")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    for name, label in mapping.items():
        print(f"{label}: {name}")
    
    # 4. ワンホットエンコーディングの適用 (CVグループ用)
    dummies = pd.get_dummies(df['balanced_group'], prefix='cat').astype(int)
    df = pd.concat([df, dummies], axis=1)
    
    # 統計情報の確認
    print("\n--- Balanced Category Statistics (Group 1-5) ---")
    for i in range(1, 6):
        col = f'cat_{i}'
        sub_df = df[df[col] == 1]
        count = len(sub_df)
        closure_rate = sub_df['is_closed'].mean()
        # 各グループ内の業態トップ3を表示して「似ているか」を確認
        top_types = sub_df['category'].value_counts(normalize=True).head(3).to_dict()
        top_types_str = ", ".join([f"{k}: {v:.1%}" for k, v in top_types.items()])
        print(f"{col}: count={count}, closure_rate={closure_rate:.4%}, composition=[{top_types_str}]")
    
    # 不要な中間カラムを削除
    df = df.drop(columns=['stratify_key'])
    
    # 保存
    df.to_csv(output_path, index=False)
    print(f"\nSaved categorized, encoded, and labeled data to {output_path}")

if __name__ == "__main__":
    main()
