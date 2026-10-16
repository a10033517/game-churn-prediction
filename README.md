# 手遊玩家流失預測流程

以玩家安裝日期與登入事件建立特徵，分開觀察期間與標籤期間，比較多數類別基準、Logistic Regression 與 Random Forest。

A player churn prediction exercise with separate feature and label windows, class-imbalance handling, and baseline/model comparison.

## 資料與適用範圍

資料來自下載腳本指定的 Kaggle `rajatsurana979/mobile-game-retention-and-churn-analysis` 資料集。尚未核實其是否來自真實產品營運，故本專案作為建模方法練習，不據此推論實際玩家流失原因或產品成效。

| 檔案 | 內容 |
|---|---|
| `users.xlsx` | 5,000 位玩家與安裝日期 |
| `events.xlsx` | 27,291 筆登入事件，事件類型為 `session_start` |
| `features.csv` | 每位玩家的觀察期間特徵與流失標籤 |

## 預測問題與時間窗

- **預測時間**：安裝後第 13 天結束時。
- **特徵期間**：安裝後第 0–13 天。
- **標籤期間**：第 14–27 天；完全沒有登入事件者標記為 `churned=1`，其餘為 `0`。

時間窗分離可避免直接把標籤期間的活動納入特徵；仍需另行檢查追蹤是否完整、前處理與驗證方式等風險，不能僅據此宣稱完全沒有資料洩漏。

## 特徵與模型

| 特徵 | 定義 |
|---|---|
| `days_active_14d` | 觀察期間有活動的天數 |
| `active_ratio_14d` | 活動天數除以 14 |
| `recency_days` | 距第 13 天最近一次活動的間隔 |
| `first_week_sessions`、`second_week_sessions` | 第一週與第二週的活動天數；欄位名稱中的 sessions 不是登入次數 |
| `trend` | 第二週活動天數減去第一週 |
| `max_gap_days` | 已觀測到的相鄰活動日最大間隔 |

使用 75%／25% 分層訓練／測試切分，固定 `random_state=42`。Logistic Regression 先以訓練集擬合 StandardScaler；兩個模型皆設定 `class_weight="balanced"`。

## 既有測試結果

全樣本流失比例為 **93.74%**（4,687／5,000），未流失玩家為 313 人。以下數值來自 [model_results.csv](model_results.csv)，正類均為 **流失 `churned=1`**。

| 模型 | ROC-AUC | Average Precision（流失正類） |
|---|---:|---:|
| 多數類別基準 | 0.5000 | 0.9376 |
| Logistic Regression | 0.9940 | 0.9993 |
| Random Forest | 0.9940 | 0.9993 |

基準的 Average Precision 對應測試集流失比例 **93.76%**。Average Precision 是 `average_precision_score` 的輸出，與以梯形法計算 PR 曲線面積的定義不同；它也不是未流失少數類別的指標。

只預測流失即可得到約 93.76% 的 accuracy，卻無法找出未流失玩家。因此評估時還需查看各類別的 precision、recall、混淆矩陣，以及以未流失為正類的指標；目前 CSV 未保存後者。

## 解讀與限制

- 兩個模型在這次切分中的結果接近，尚未以重複切分或獨立資料驗證穩定性。
- 高分不足以確認資料真實性、模型可泛化性或不存在洩漏；也不能單憑分數或玩家人數判定資料一定為模擬生成。
- 資料僅含登入事件，缺少營收、關卡、社交與裝置等面向。
- 「14 天未登入」是本專案操作定義，不等於永久流失。
- 尚未做時間外測試、機率校準、成本導向門檻評估或線上介入實驗。

後續可先確認資料來源與追蹤完整性，再補上少數類別指標、信賴區間及時間外驗證，評估是否適合更接近實際營運的場景。

## 檔案

- [01_get_data.py](01_get_data.py)：取得資料，需要自行設定 Kaggle 憑證。
- [02_build_features.py](02_build_features.py)：特徵與標籤生成。
- [03_train_model.py](03_train_model.py)：基準、模型訓練與評估。
- [model_results.csv](model_results.csv)：既有模型摘要。
- [reports](reports/)：延伸分析文件。

## 重現分析

在專案根目錄執行，以下為 Windows PowerShell 範例：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe 02_build_features.py
.venv\Scripts\python.exe 03_train_model.py
```

以上使用專案已有的 Excel 檔。若缺少原始檔，先完成 Kaggle 憑證設定，再執行 `.venv\Scripts\python.exe 01_get_data.py`。重跑會更新特徵與結果 CSV。
