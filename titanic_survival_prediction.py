import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#データの読み込み
try:
    # データをDataFrameとして読み込み
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_submission = pd.read_csv('gender_submission.csv') # 提出フォーマット
    
    # データの確認
    print("\n--- Train Data Head ---")
    print(df_train.head())
    print("\n--- Test Data Head (Survived列がないことを確認) ---")
    print(df_test.head())
    
except FileNotFoundError:
    print("\nエラー: 'train.csv', 'test.csv', 'gender_submission.csv' のいずれかのファイルが見つかりません。")
    print("Kaggleからダウンロードして、このスクリプトと同じディレクトリに配置してください。")
    exit()


# データの結合と処理
# 訓練データとテストデータを結合し、一括で前処理を行います。
# Survived列を訓練データから分離し、テストデータには存在しないことを確認します。
target = df_train['Survived']
df_train.drop('Survived', axis=1, inplace=True)

# 結合のための準備
# train/testの区別をつけるためにフラグ列を追加（必須ではありませんが慣例）
df_train['is_test'] = 0
df_test['is_test'] = 1

# データの結合
df_combined = pd.concat([df_train, df_test], ignore_index=True)

print("\n--- 結合されたデータの欠損値カウント ---")
print(df_combined.isnull().sum()[df_combined.isnull().sum() > 0])

# 3-2: 欠損値の処理（補完）
# Age (年齢): 中央値で補完するのが一般的です。
df_combined['Age'].fillna(df_combined['Age'].median(), inplace=True)

# Fare (運賃): 欠損値はテストデータに1つだけあります。中央値で補完します。
df_combined['Fare'].fillna(df_combined['Fare'].median(), inplace=True)

# Embarked (乗船港): 最も多い値（最頻値）で補完します。
df_combined['Embarked'].fillna(df_combined['Embarked'].mode()[0], inplace=True)

# 3-3: カテゴリカル変数のエンコーディング（数値化）
# Sex (性別): 'male'を0, 'female'を1に変換します。（バイナリエンコーディング）
df_combined['Sex'] = df_combined['Sex'].map({'male': 0, 'female': 1})

# Embarked (乗船港): One-Hotエンコーディングを行います。
df_combined = pd.get_dummies(df_combined, columns=['Embarked'], prefix='Embarked')

# 3-4: 不要な列の削除
# 予測に直接役立たない、またはデータが不揃いな列を削除します。
# Name, Ticket, Cabinは、より高度な特徴量エンジニアリングが必要なため、ここでは削除します。
df_combined.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

print("\n--- 前処理後のデータセットの列 ---")
print(df_combined.columns)


# ----------------------------------------------------------------------
# ステップ4: 訓練データとテストデータに再分割し、モデルを構築・学習
# ----------------------------------------------------------------------

# 4-1: データの再分割
# 処理後の結合データから元の訓練用とテスト用に分割します。
X_train = df_combined[df_combined['is_test'] == 0].drop('is_test', axis=1)
X_test = df_combined[df_combined['is_test'] == 1].drop('is_test', axis=1)

# X_trainとX_testから欠損値処理で作成したEmbarked_S, Embarked_C, Embarked_Q列が削除されてないか確認
print("\n--- 最終的な特徴量の確認 ---")
print(X_train.head())

# 4-2: モデルの構築と学習（ロジスティック回帰）
# ロジスティック回帰モデルを選択します。分類問題のベースラインとして最適です。
model = LogisticRegression(solver='liblinear', random_state=42)

# モデルの学習
model.fit(X_train, target)

print("\nモデルの学習が完了しました。(ロジスティック回帰)")

# ----------------------------------------------------------------------
# ステップ5: 予測と提出ファイルの作成
# ----------------------------------------------------------------------

# 5-1: テストデータで予測
predictions = model.predict(X_test)

# 5-2: 提出ファイルの作成
# PassengerIdと予測結果（Survived）を含むDataFrameを作成します。
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'], # 元のテストデータからPassengerIdを取得
    'Survived': predictions
})

# Survived列は整数型（0または1）である必要があります
submission['Survived'] = submission['Survived'].astype(int)

# 提出用CSVファイルとして保存
submission.to_csv('submission_baseline.csv', index=False)

print("\n--- 予測結果の提出ファイルを作成しました ---")
print("ファイル名: submission_baseline.csv")
print(submission.head())

print("\nこのファイルをKaggleに提出することで、他のKagglerと比較したランキングを確認できます。")
# ----------------------------------------------------------------------
