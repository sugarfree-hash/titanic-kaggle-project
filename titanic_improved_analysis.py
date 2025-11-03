# ----------------------------------------------------------------------
# Kaggle Titanic Survival Prediction: Deck特徴量追加版
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier

print("必要なライブラリのインポートが完了しました。")

# ----------------------------------------------------------------------
# ステップ1: データの読み込みと結合 (前回のコードから変更なし)
# ----------------------------------------------------------------------
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError:
    print("\nエラー: 'train.csv', 'test.csv' のファイルが見つかりません。")
    exit()

target = df_train['Survived']
df_train.drop('Survived', axis=1, inplace=True)

df_train['is_test'] = 0
df_test['is_test'] = 1
df_combined = pd.concat([df_train, df_test], ignore_index=True)

print("\n--- データ結合完了 ---")

# ----------------------------------------------------------------------
# ステップ2: 高度な特徴量エンジニアリング
# ----------------------------------------------------------------------

# 2-1: Title (敬称) の抽出と統合 (前回と同じ)
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

df_combined['Title'] = df_combined['Name'].apply(get_title)
rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df_combined['Title'] = df_combined['Title'].replace(rare_titles, 'Rare')
df_combined['Title'] = df_combined['Title'].replace('Mlle', 'Miss')
df_combined['Title'] = df_combined['Title'].replace('Ms', 'Miss')
df_combined['Title'] = df_combined['Title'].replace('Mme', 'Mrs')


# 2-2: FamilySize (家族人数) の作成と分類 (前回と同じ)
df_combined['FamilySize'] = df_combined['SibSp'] + df_combined['Parch'] + 1
df_combined['IsAlone'] = np.where(df_combined['FamilySize'] == 1, 1, 0)


# 2-3: 【新規】Deck (デッキ) 情報の作成
# 欠損値を 'M' (Missing) で補完し、先頭1文字を抽出してDeckとする。
df_combined['Cabin'].fillna('M', inplace=True)
df_combined['Deck'] = df_combined['Cabin'].str[0] 


# ----------------------------------------------------------------------
# ステップ3: 欠損値の処理とエンコーディング
# ----------------------------------------------------------------------

# 3-1: 数値データの欠損値補完
df_combined['Age'].fillna(df_combined['Age'].median(), inplace=True)
df_combined['Fare'].fillna(df_combined['Fare'].median(), inplace=True)
df_combined['Embarked'].fillna(df_combined['Embarked'].mode()[0], inplace=True)


# 3-2: カテゴリカル変数のOne-Hotエンコーディング
# Pclass, Sex, Embarked, Title に加えて、新しく 'Deck' を追加！
features_to_encode = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
df_combined = pd.get_dummies(df_combined, columns=features_to_encode)


# 3-3: 不要な列の削除
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
df_combined.drop(cols_to_drop, axis=1, inplace=True)


# ----------------------------------------------------------------------
# ステップ4 & 5: モデルの学習、予測、提出ファイルの作成 (前回と同じ)
# ----------------------------------------------------------------------

X_train = df_combined[df_combined['is_test'] == 0].drop('is_test', axis=1)
X_test = df_combined[df_combined['is_test'] == 1].drop('is_test', axis=1)

# ランダムフォレストモデル
model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
model.fit(X_train, target)

print("\nモデルの学習が完了しました。(Deck特徴量追加版)")

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': predictions
})

submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission_deck_feature.csv', index=False)

print("\n--- 予測結果の提出ファイルを作成しました ---")
print("ファイル名: submission_deck_feature.csv")
print("このファイルをKaggleに提出し、スコアの向上を確認しましょう！")
