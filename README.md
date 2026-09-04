# 手遊玩家流失預測（端對端流失預測模型）

## 動機

前兩個專案（Free Fire 評論分析、Cookie Cats A/B 測試）分別練了文字分析與統計檢定；這個專案要把範圍拉大成一個**完整的機器學習流程**：特徵工程 → 處理類別不平衡 → 訓練與比較模型 → 評估 → 轉成商業建議，同時刻意示範怎麼避開 Cookie Cats 專案遇到的**資料洩漏（data leakage）**問題。

## ⚠️ 資料來源說明（請務必先讀這段）

這份資料來自 Kaggle 的 `rajatsurana979/mobile-game-retention-and-churn-analysis`。我**沒有辦法確認這是真實產品的營運資料，還是模擬產生的資料**——查證過程中找到一篇研究論文提到「一份包含 5,000 位玩家的真實感模擬資料集（realistically simulated dataset of 5,000 players）」，人數剛好與本資料集的 5,000 位玩家吻合，加上資料本身的乾淨程度（只有單一事件類型、沒有任何缺值或雜訊）也不太像真實產品日誌會有的樣子，種種跡象都指向**這很可能是模擬資料，不是真實玩家行為**。

這不影響拿它來練習完整的機器學習流程（特徵工程、處理不平衡、訓練評估），但**不該拿這個專案的結論宣稱是「真實玩家的流失原因」**——這點在 README、履歷、面試時都要講清楚，避免誇大資料的真實性。這也是為什麼下面的模型結果好到有點可疑時，本文會直接把「這可能反映模擬資料過於乾淨」寫成一個正式的分析發現，而不是隱藏起來。

## 資料

- `users.xlsx`：5,000 位玩家，含安裝日期（install_date），涵蓋 2024-01-01 至 2024-01-30
- `events.xlsx`：27,291 筆 `session_start` 事件（玩家當天有登入遊戲），涵蓋 2024-01-01 至 2024-03-05

## 方法：如何避開資料洩漏

Cookie Cats 專案的限制之一，是 `sum_gamerounds` 這個欄位涵蓋安裝後 14 天的局數，但用它去預測「Day-7 留存」時，其實混入了 Day-7 之後才發生的資訊，構成洩漏。這個專案刻意用時間窗把「特徵」和「標籤」隔開：

- **特徵窗**：每位玩家安裝後第 0–13 天的活動，只用這段期間的資料算特徵
- **標籤窗**：安裝後第 14–27 天，這段期間**完全沒有** `session_start` 事件的玩家，標記為流失（`churned=1`）
- 兩個時間窗不重疊，預測當下（第 13 天結束）不可能看到標籤窗發生的事，沒有洩漏

### 特徵工程（`02_build_features.py`）

| 特徵 | 說明 |
|---|---|
| `days_active_14d` | 特徵窗內有活動的天數 |
| `active_ratio_14d` | 上述天數 ÷ 14 |
| `recency_days` | 距離特徵窗結束，最後一次活動已經過幾天（愈大代表愈久沒玩） |
| `first_week_sessions` / `second_week_sessions` | 第 1 週（第0–6天）與第 2 週（第7–13天）的活動天數，拆開看是否有「愈玩愈少」的趨勢 |
| `trend` | 第2週−第1週，正值代表活動量成長，負值代表衰退 |
| `max_gap_days` | 特徵窗內連續兩次活動間隔的最大天數 |

### 模型（`03_train_model.py`）

- 資料切分：75/25 train/test，依 `churned` 分層抽樣（stratified）
- 先跑一個**永遠猜多數類別**的 naive baseline，作為比較基準——這一步很重要，因為流失率高達 93.7%，光看 accuracy 會被誤導
- Logistic Regression 與 Random Forest 都設定 `class_weight="balanced"`，避免模型單純學會「猜流失就好」

## 結果

**類別分布**：流失率 93.74%（5,000 人中僅 313 人在第14–27天仍有活動）

**Naive baseline（永遠猜「流失」）**：accuracy 93.8%，但對「留存」這個類別的 precision/recall 都是 0——這就是為什麼**accuracy 在極度不平衡的資料上是誤導性指標**，必須看 precision/recall/F1，尤其是少數類別（留存玩家）的表現。

| 模型 | ROC-AUC | PR-AUC（Average Precision） | 留存類別 Precision | 留存類別 Recall |
|---|---|---|---|---|
| Naive baseline | 0.500 | 0.063（=流失率） | 0.000 | 0.000 |
| Logistic Regression | 0.994 | 0.999 | 0.848 | 1.000 |
| Random Forest | 0.994 | 0.999 | 0.848 | 1.000 |

兩個模型表現幾乎一致，且**好到不太真實**——ROC-AUC 0.994 在真實世界的流失預測任務中非常罕見。最重要的特徵是 `days_active_14d`、`trend`、`second_week_sessions`：簡單說，「特徵窗第二週還在玩的人，之後也大機率繼續玩」，這個關係在這份資料裡幾乎是決定性的，這正好呼應上面「資料很可能是模擬產生」的懷疑——真實世界的玩家行為通常雜訊更多，不會有這麼乾淨的決定性關係。

## 討論

如果我是負責這個專案的分析師，看到 0.994 的 AUC，**第一反應不會是慶祝，而是懷疑**：多半代表資料太乾淨、規則太簡單，或標籤定義本身讓答案幾乎寫在特徵裡（雖然這裡已經刻意做了時間窗隔離，排除了嚴格意義上的洩漏）。這是這個專案除了「跑完一個 ML pipeline」之外，最想留下的心得：**模型表現好到反常時，該懷疑的是資料，不是急著上線部署。**

## 限制

- 資料極可能是模擬產生，結論不能代表真實玩家行為，僅作為方法論練習
- 只有單一事件類型（`session_start`），沒有付費、關卡進度、社交等真實產品常見的行為維度，特徵工程的深度因此受限
- 流失率高達 93.7%，樣本中留存玩家僅 313 人，測試集中留存玩家只有 78 人，模型評估結果的信賴區間本身也會很寬，小樣本下的高 AUC 需要更謹慎看待
- 未使用交叉驗證（僅單次 train/test split），嚴謹的作法應該用 k-fold 交叉驗證取得更穩定的效能估計

## 檔案結構

```
game-churn-prediction/
├── 01_get_data.py          # 從 Kaggle 下載資料（需要自己的 Kaggle API token）
├── 02_build_features.py    # 特徵工程（含防洩漏的時間窗設計）
├── 03_train_model.py       # 訓練與評估模型
├── requirements.txt
├── users.xlsx / events.xlsx  # 原始資料
├── features.csv             # 特徵表
└── model_results.csv        # 模型比較結果
```

## 如何重現

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# 需要先在 https://www.kaggle.com/settings 建立自己的 API token
python 01_get_data.py
python 02_build_features.py
python 03_train_model.py
```
