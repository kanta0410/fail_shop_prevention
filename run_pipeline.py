"""
廃業予測モデル - マスター実行スクリプト
要件定義書の全ステップを順番に実行する。
"""
import subprocess
import sys
import os

STEPS = [
    {
        'name': 'Step4-E: グループA・B・E 特徴量エンジニアリング',
        'cmd': [sys.executable, 'src/preprocessing/build_features.py']
    },
    {
        'name': 'Step5-9: モデル学習・評価・SHAP分析',
        'cmd': [sys.executable, 'src/models/train_pipeline.py']
    },
    {
        'name': 'Step10: 廃業リスクヒートマップ生成',
        'cmd': [sys.executable, 'src/models/generate_heatmap.py']
    }
]

def run_step(name, cmd):
    print(f"\n{'=' * 60}")
    print(f"Executing: {name}")
    print('=' * 60)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR in {name}! Stopping.")
        sys.exit(1)
    print(f"Completed: {name}")

if __name__ == '__main__':
    for step in STEPS:
        run_step(step['name'], step['cmd'])
    print("\nAll steps completed successfully!")
