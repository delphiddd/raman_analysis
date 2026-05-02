# Raman/SERS THC Detection

Notebook สำหรับวิเคราะห์ Raman/SERS spectra เพื่อตรวจหา THC แบบ binary classification (No THC = 0, THC = 1) และทดลอง threshold ความเข้มข้น

## ไฟล์

- `raman_analysis_def.ipynb` — notebook หลัก ครอบคลุมตั้งแต่ data loading → preprocessing → PCA → classification

## Pipeline

1. **Data loading** — `load_raman_recursive()` อ่านไฟล์ `.txt` แบบ recursive จาก `Datasets/` (ใช้ชื่อโฟลเดอร์ย่อยเป็น label)
2. **Preprocessing** — drPLS baseline (`pybaselines`) → Savitzky–Golay smoothing → SNV normalization → SNR filter
3. **Feature selection** — เลือก 1 ใน 3 วิธี
   - `crop_spectra()` ตัดช่วง 620–1700 cm⁻¹ (full spectra)
   - `select_peaks()` ดึง 9 peaks: `[634, 645, 683, 769, 1100, 1155, 1222, 1495, 1624]` cm⁻¹
   - `sersmap()` ยุบทุก 25 spectra เป็น 1 (mean) สำหรับ SERS map
4. **Dimensionality reduction** — PCA 2D / 3D
5. **Classification** — KFold leave-25-out กับ 8 โมเดล: Logistic Regression, LDA, Extra Trees, Gradient Boosting, Decision Tree, Random Forest, Ridge Classifier, SVM (RBF)

## Datasets ที่ใช้

- `Cal. curve (10-10-12)/` — calibration curve (มี `Concentration`)
- `Selectivity/` — ทดสอบความจำเพาะกับสารอื่น (THC, CBD, CBN ฯลฯ)
- `Solvents/` — ตัวทำละลาย (4-AP, ACN, FBBB)

> หมายเหตุ: โฟลเดอร์ `Datasets/` ไม่ได้รวมไว้ใน repo

## การทดลอง Threshold

- `df_cal_Threshold` — แบ่งที่ **6.4 ppm** (`less6.4` / `more6.4`)
- `df_cal_Threshold_32` — แบ่งที่ **3.2 ppm** (`less3.2` / `more3.2`)

## Dependencies

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pybaselines`, `scipy`
