import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# ファイルパス（data/フォルダを使用）
train_path = 'data/train.csv'
test_path = 'data/test.csv'
submission_output_path = 'data/submission_group_feature.csv'

# データ読み込み
try:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path) 
    
    # 予測提出に必要なIDを保持
    test_passenger_ids = test_data['PassengerId'].copy() 

except FileNotFoundError:
    print(f"\nエラー: 必須のデータファイルが見つかりません: {e.filename}")
    exit()

# 訓練データとテストデータを結合
df_combined = pd.concat([train_data.drop('Survived', axis=1), test_data], sort=False)

print("\n--- データ結合完了。特徴量エンジニアリング開始 ---")

# -----
# 1. 基礎特徴量の作成と整形
# -----

# 1-1. Title (敬称) の抽出と整形
df_combined['Title'] = df_combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df_combined['Title'] = df_combined['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_combined['Title'] = df_combined['Title'].replace('Mlle', 'Miss')
df_combined['Title'] = df_combined['Title'].replace('Ms', 'Miss')
df_combined['Title'] = df_combined['Title'].replace('Mme', 'Mrs')

# 1-2. Deck (デッキ) の抽出
df_combined['Cabin'].fillna('M', inplace=True)
df_combined['Deck'] = df_combined['Cabin'].str[0]
df_combined.drop('Cabin', axis=1, inplace=True)

# 1-3. FamilySize (家族サイズ) の作成
df_combined['FamilySize'] = df_combined['SibSp'] + df_combined['Parch'] + 1
df_combined.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# 1-4. GroupSize (チケットごとの乗客数)
ticket_counts = df_combined['Ticket'].value_counts()
df_combined['GroupSize'] = df_combined['Ticket'].map(ticket_counts)


# -----
# 2. GroupSurvivalRate (同行者の生存率) の計算
# -----

# 訓練データとテストデータを一旦分離
train_only = df_combined.iloc[:len(train_data)].copy()
test_only = df_combined.iloc[len(train_data):].copy()
train_only['Survived'] = train_data['Survived']

# Ticketグループごとの平均生存率を計算 (自分自身を含む)
group_survival_mean = train_only.groupby('Ticket')['Survived'].mean()

# 訓練データにマッピング
train_only['GroupSurvivalRate'] = train_only['Ticket'].map(group_survival_mean)

# 訓練データ: 自分自身の影響を除外した生存率を計算 (ターゲットリーク防止)
# 計算式: (グループ平均 * グループサイズ - 自分の生存) / (グループサイズ - 1)
train_only['GroupSurvivalRate'] = (train_only['GroupSurvivalRate'] * train_only['GroupSize'] - train_only['Survived']) / (train_only['GroupSize'] - 1)

# テストデータ: 訓練データで計算したグループ平均をそのままマッピング
test_only['GroupSurvivalRate'] = test_only['Ticket'].map(group_survival_mean)

# 再度結合
df_combined['GroupSurvivalRate'] = pd.concat([train_only['GroupSurvivalRate'], test_only['GroupSurvivalRate']])

# -----
# 3. 欠損値の最終補完とエンコーディング
# -----

# 数値データの欠損値補完 (単独乗船者や NaN の Age/Fare/Embarked)
df_combined['GroupSurvivalRate'].fillna(df_combined['GroupSurvivalRate'].median(), inplace=True)
df_combined['Age'].fillna(df_combined['Age'].median(), inplace=True)
df_combined['Fare'].fillna(df_combined['Fare'].median(), inplace=True)
df_combined['Embarked'].fillna(df_combined['Embarked'].mode()[0], inplace=True)

# Pclass, Sex, Embarked, Title, Deck をOne-Hotエンコーディング
features_to_encode = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
df_combined = pd.get_dummies(df_combined, columns=features_to_encode, drop_first=True)

# 不要な列の削除
df_combined.drop(['Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# データセットの分離
X_train = df_combined.iloc[:len(train_data)]
X_test = df_combined.iloc[len(train_data):]
target = train_data['Survived']

# -----
# 4. モデル訓練と予測
# -----

# モデル設定と訓練
# 1. ベースモデルの定義 (過去の経験やチューニング結果を元に最適なパラメータを設定)
# モデル1: ランダムフォレスト (安定性と汎用性)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

# モデル2: XGBoost (高精度と非線形性の学習)
xgb_model = XGBClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=5,
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

# 2. 投票分類器の定義
# 'hard' voting は、個々のモデルの予測結果の多数決を採用します。
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)], 
    voting='hard'
)

# 3. 訓練の実行 (アンサンブル全体を訓練)
voting_clf.fit(X_train, target)
model = voting_clf

# 予測の実行
predictions = model.predict(X_test)

# 提出ファイル形式に整形
submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
submission['Survived'] = submission['Survived'].astype(int)

# CSVファイルの出力
submission.to_csv(submission_output_path, index=False)

print(f"\n提出ファイル '{submission_output_path}' が正常に生成されました。")
print(f"予測結果の最初の5行:\n{submission.head()}")