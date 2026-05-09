# `model.ipynb` — เอกสารอธิบายโดยละเอียด

ไฟล์นี้สรุป pipeline ทั้งหมดของ `model.ipynb` ตั้งแต่การโหลดข้อมูล Raman/SERS, preprocessing, การเลือก feature, การ cross-validation ด้วย classifier 8 ตัว, ไปจนถึง threshold experiment สำหรับ THC detection

---

## ภาพรวม Pipeline

```
Raw .txt files (X, Y, Wave, Intensity)
        │
        ▼
[1] load_raman_recursive  ─────► concat ทุกไฟล์ → DataFrame เดียว, แปะ label จาก folder name
        │
        ▼
[2] encode_df             ─────► สร้างคอลัมน์ Target (No THC / THC) + Target_Code (0 / 1)
        │
        ▼
[3] remove_duplicates     ─────► long format → matrix (n_samples × n_wavenumbers)
        │
        ▼
[4] preprocessing         ─────► drPLS baseline → SavGol smoothing → SNR cutoff → SNV
        │
        ▼
[5] crop_spectra OR select_peaks OR sersmap   (3 feature views)
        │
        ▼
[6] loovc / ser_loovc     ─────► Leave-25-out (หรือ Leave-One-Out) × 8 classifiers
        │                       + permutation feature importance + confusion matrices
        ▼
   Summary DataFrame + กราฟ
```

---

## รายละเอียดแต่ละฟังก์ชัน

### 1. `load_raman_recursive(base_path, label_col_name='Label')`

**หน้าที่**: โหลดไฟล์ `.txt` ทุกไฟล์ใต้ `base_path` (รวม subfolder) แล้วรวมเป็น DataFrame เดียว

**การทำงาน**:
1. `glob('**/*.txt', recursive=True)` ค้นหาไฟล์ทั้งหมด → `sorted()` เพื่อลำดับคงที่
2. อ่านแต่ละไฟล์ด้วย `pd.read_csv(delimiter='\t', names=['X','Y','Wave','Intensity'], skiprows=1)`
3. แปะ label จากชื่อ folder ที่ไฟล์อยู่: `os.path.basename(os.path.dirname(filepath))`
4. แปะ `Filename` ไว้ตามรอย sample
5. `try/except` ครอบไว้ — ไฟล์ที่อ่านไม่ได้จะข้าม
6. ตัด `X`, `Y` (พิกัด stage) ทิ้งก่อน return

**Input**: path string + ชื่อ label column (default `'Label'`)
**Output**: DataFrame คอลัมน์ `Wave`, `Intensity`, `<label_col_name>`, `Filename`

**ใช้ตอนไหน**: cell แรกของ pipeline หลัก โหลด 3 dataset (Cal. curve, Selectivity, Solvents)

---

### 2. `encode_df(df_cal, df_select, df_solv)`

**หน้าที่**: แปลง label ที่ต่างกันใน 3 dataset ให้เป็น binary classification เดียวกัน (No THC = 0 / THC = 1)

**กฎการ encode**:
| Dataset | Source column | กฎ |
|---|---|---|
| `df_cal` | `Concentration` | `0 ppm → No THC`, อื่น → `THC` |
| `df_select` | `Substance` | substring `'THC'` ในชื่อ → `THC`, ไม่งั้น → `No THC` |
| `df_solv` | `Solvent_Type` | substring `'0'` → `No THC`, ไม่งั้น → `THC` |

**ผลลัพธ์**: เพิ่มคอลัมน์ `Target` (string) และ `Target_Code` (int 0/1) ใน-place ทั้ง 3 DataFrame

**สำคัญ**: เช็ค `df.empty` ก่อนทุกครั้ง — ป้องกันโยน error เมื่อ dataset ไหนยังไม่โหลด

---

### 3. `remove_duplicates(df)` — Long → Matrix Reshape

**หน้าที่**: เปลี่ยนข้อมูล long format (`Wave`/`Intensity` เรียงต่อกันยาว ๆ) ให้เป็น matrix `(n_samples, n_wavenumbers)`

**ขั้นตอน**:
1. ดึง `wave_raw`, `intensity_raw`, `target_raw` เป็น numpy array
2. **ตรวจทิศทาง wavenumber** ด้วย `np.diff(wave_raw)`:
   - หา `first_step` ตัวแรกที่ ≠ 0
   - ถ้า > 0 → เรียงน้อยไปมาก → reset point คือจุดที่ `diff < 0`
   - ถ้า < 0 → เรียงมากไปน้อย → reset point คือจุดที่ `diff > 0`
3. **คำนวณ period**: `n_wav = resets[0] + 1` (จำนวน wavenumber ต่อ 1 spectrum)
4. Slice/reshape:
   - `wavenumbers = wave_raw[:n_wav]` — แกน X เก็บแค่ block แรก
   - `intensities = intensity_raw.reshape(-1, n_wav)` — matrix
   - `Y_labels = target_raw[::n_wav]` — เก็บ label ตัวเดียวต่อ spectrum
5. **Branching ตาม column ที่มี**:
   - มี `Concentration` → return `(wave, intensity, Y, Conc)`
   - มี `Substance` → return `(wave, intensity, Y, sub)`
   - ไม่มีทั้งคู่ → return `(wave, intensity, Y)`

**กฎสำคัญ**: ห้าม assume ทิศทาง wavenumber — ใช้ `np.diff` ตรวจทุกครั้ง

---

### 4. `preprocessing(intensity, Wave, Y_labels, Concentrations=None, snr_threshold=0.5)`

**หน้าที่**: chain 4 ขั้นตอน preprocessing + คัด outlier ด้วย SNR

**Pipeline (ทำทีละ spectrum):**
1. **Baseline correction** — `Baseline(x_data=Wave).drpls(spectrum, lam=1e6, eta=0.5)` (drPLS algorithm)
2. **Subtraction** — `baseline_corrected = spectrum - baseline`
3. **Smoothing** — `savgol_filter(window_length=15, polyorder=2)`
4. **SNR calculation** — `np.max(smoothed) / (np.std(smoothed) + 1e-9)` (เก็บไว้ใน `snr_list`)
5. **SNV normalization** — `(smoothed - mean) / (std + 1e-9)` → เก็บลง `X_processed`

**Outlier removal**:
- หลังลูปจบ → `good_indices = snr_array > snr_threshold` (default 0.5)
- ใช้ mask **เดียวกัน** คัดทุก array (intensity, Y, Concentration) พร้อมกัน — กัน array หลุด alignment

**Return**:
- ถ้ามี `Concentrations` → `(intensity_final, Y_final, Conc_final)`
- ถ้าไม่มี → `(intensity_final, Y_final)`

**กฎสำคัญ**: SNV normalize มาแล้ว → **ห้ามใส่ StandardScaler ก่อน classifier อีก**

---

### 5. `crop_spectra(intensities, wavenumbers, min_wave=620, max_wave=1700)`

**หน้าที่**: ตัดช่วง wavenumber ที่สนใจ (Full Spectra view)

**การทำงาน**:
1. สร้าง mask: `(wavenumbers >= min_wave) & (wavenumbers <= max_wave)`
2. Apply mask ทั้งแกน X และ Y พร้อมกัน
3. Return `(Wave_cropped, intensity_cropped)`

**ค่า default**: 620–1700 cm⁻¹ (fingerprint region สำหรับ THC)

---

### 6. `select_peaks(X_clean, wavenumbers, target_peaks=None)`

**หน้าที่**: เลือกเฉพาะ feature ที่ index ใกล้ peak ที่กำหนด (Selected Peaks view)

**การทำงาน**:
1. วน loop peak ที่อยากได้
2. หา index ที่ใกล้ที่สุด: `np.argmin(np.abs(wavenumbers - p))`
3. เก็บ index + ค่า wavenumber จริงที่เจอ (อาจไม่ตรงเป๊ะ)
4. Slice column: `X_selected = X_clean[:, peak_indices]`

**Peak set ที่ใช้ในงาน THC** (ค่อย ๆ ablate):
- 9 peaks: `[634, 645, 683, 769, 1100, 1155, 1222, 1495, 1624]`
- 8 peaks: ตัด `1222` ออก
- 7 peaks: ตัด `1155` ออกเพิ่ม
- 6 peaks (final): `[634, 645, 683, 1100, 1495, 1624]`

---

### 7. `sersmap(intensity_final, Y_label, Conc=None)`

**หน้าที่**: เฉลี่ย 25 spectra (5×5 SERS map) ต่อ 1 จุด measurement → ลด noise

**การทำงาน**:
1. อ่าน `n_wav` จาก `shape[1]` (ไม่ hard-code)
2. Reshape: `(-1, 25, n_wav).mean(axis=1)` — เฉลี่ย 25 spectra ในแต่ละกลุ่ม
3. Resize Y: `Y.reshape(-1, 25)[:, 0]` (label ใน 25 spectra เหมือนกันอยู่แล้ว เลือกตัวแรกได้)
4. ถ้ามี `Conc` → resize ด้วยวิธีเดียวกัน

**ใช้คู่กับ**: `ser_loovc()` (Leave-One-Out CV) เพราะหลังเฉลี่ยแล้ว sample เหลือน้อย

---

### 8. `loovc(X_train, Y_train, class_name=None, wavenumbers=None, compute_importance=True, n_repeats=30, top_k=20)`

**หน้าที่**: ตัวหลักของการประเมินโมเดล — Leave-25-out CV + 8 classifiers + permutation importance

**ขั้นตอน**:

#### A. Setup KFold
- `n = len(X_train) // 25` — จำนวน fold
- `KFold(n_splits=n)` — **ไม่ shuffle** เพื่อรักษาลำดับ measurement (`shuffle=False` โดย implicit)

#### B. นิยาม 8 Classifiers
| Model | สร้างด้วย |
|---|---|
| Logistic Regression | `LogisticRegression(n_jobs=-1)` |
| LDA | `LinearDiscriminantAnalysis()` |
| Extra Trees | `ExtraTreesClassifier(n_jobs=-1)` |
| Gradient Boosting | `GradientBoostingClassifier()` |
| Decision Tree | `DecisionTreeClassifier()` |
| Random Forest | `RandomForestClassifier(n_jobs=-1)` |
| Ridge Classifier | `RidgeClassifier()` |
| SVM (RBF) | `SVC(kernel='rbf')` |

#### C. CV Loop
- ทุก fold:
  - `X_train_final / X_test_final / y_train_final / y_test_final` แยกตาม index
  - แต่ละ classifier ห่อใน `Pipeline([('classifier', model)])` (ไม่มี scaler)
  - `fit → predict` → คำนวณ `accuracy_score`, `precision_score(zero_division=0)`, `f1_score(zero_division=0)`
  - สะสม confusion matrix: `cm_results[name] += confusion_matrix(..., labels=[0,1])`
- Print progress ทุก ๆ 10 รอบ

#### D. Summary
- เฉลี่ย metric ทุก fold → `summary_df` (DataFrame)
- เก็บ raw metric per fold → `summary_raw` (สำหรับวิเคราะห์ variance)
- `display(summary_df)` ใน Jupyter

#### E. Confusion Matrices
- เรียก `plot_all_confusion_matrices(cm_results, class_name)` พล็อตรวม 2×4

#### F. Permutation Feature Importance
- **ทำหลัง CV จบ ครั้งเดียว บน full data** (ไม่ทำใน fold loop เพื่อความเร็ว + เสถียร)
- `pipe.fit(X_train, Y_train)` → `permutation_importance(pipe, X_train, Y_train, n_repeats=30, random_state=42, n_jobs=-1)`
- เก็บ `importances_mean` ของแต่ละโมเดล
- เรียก `plot_feature_importance(...)`

**Return**: `(metrics_results, summary_df, feature_importance_results, summary_raw)`

---

### 9. `plot_feature_importance(feature_importance_results, wavenumbers=None, top_k=20)`

**หน้าที่**: พล็อต permutation importance รวม 8 โมเดลใน figure เดียว (grid 2×4)

**การทำงาน**:
1. `subplots(2, 4, figsize=(24, 10), dpi=120)` → `axes.flatten()`
2. กรณีมี `wavenumbers`:
   - `ax.vlines(wavenumbers, 0, importances)` — เส้นตั้งฉากแบบ stem plot
   - หา top-K ด้วย `np.argsort(importances)[-top_k:]` → จุดแดง + annotate ตัวเลข cm⁻¹
   - แสดงเฉพาะ annotation ที่ `y_val > 0.001` กัน label เบียด
3. กรณีไม่มี `wavenumbers` → fallback bar chart `ax.barh(...)`
4. ถ้าโมเดลไม่ถึง 8 ตัว → `fig.delaxes(...)` ลบ subplot ว่าง
5. `tight_layout()` + `plt.show()`

**Output**: 1 figure, 8 subplot (subplot ละ 1 model)

---

### 10. `plot_all_confusion_matrices(cm_results, class_name=None)`

**หน้าที่**: พล็อต confusion matrix สะสม (รวมทุก fold) ของ 8 โมเดลใน figure เดียว

**การทำงาน**:
1. `subplots(2, 4, figsize=(20, 10))` → `axes.flatten()`
2. `ConfusionMatrixDisplay(cm, display_labels=class_name).plot(ax=axes[i], cmap='Blues', colorbar=False)`
3. ลบ subplot ว่างถ้ารันโมเดลไม่ถึง 8 ตัว
4. `tight_layout()` + `plt.show()`

**หมายเหตุ**: ปิด `colorbar` เพื่อกัน figure แคบ

---

### 11. `apply_pca(data, n_components=2)`

**หน้าที่**: fit PCA + report % explained variance

**การทำงาน**:
1. `pca = PCA(n_components=n_components)`
2. `X_train_pca = pca.fit_transform(data)`
3. Print `explained_variance_ratio_ * 100` (ถ้า n=2 → PC1, PC2; ถ้า n=3 → PC1, PC2, PC3)
4. Return `(X_train_pca, pca)`

**ข้อควรระวัง**: PCA ตัวนี้ **fit ทั้งชุด** — ใช้สำหรับ visualization เท่านั้น **ห้าม** เอา PC ไปป้อน classifier ตรง ๆ (data leakage)

---

### 12. `plot_pca_results(X_pca, Y_labels, pca_model, title_name="Dataset", legend_title="Label")`

**หน้าที่**: scatter plot PC1 vs PC2

**การทำงาน**:
1. ใส่ PC + label ลง DataFrame ชั่วคราว
2. `sns.scatterplot(x='PC1', y='PC2', hue='Target', palette=['#1f77b4', '#ff7f0e'], s=80, edgecolor='k')`
3. Label แกนพร้อม % explained variance: `f'PC1 ({explained_variance[0]:.2f}%)'`
4. `grid(linestyle='--', alpha=0.6)`

---

### 13. `ser_loovc(X_train, Y_train, class_name=None)`

**หน้าที่**: เวอร์ชันลดรูปของ `loovc` สำหรับ SERS-averaged data (sample น้อย)

**ต่างจาก `loovc` อย่างไร**:
| | `loovc` | `ser_loovc` |
|---|---|---|
| KFold | `n_splits = len(X)//25` | `n_splits = len(X)` (Leave-One-Out) |
| Permutation importance | ใช่ | ไม่ |
| Return | 4 ค่า | 2 ค่า (`metrics_results, summary_df`) |
| LogReg `max_iter` | default | `2000` (กัน convergence warning) |

ส่วนที่เหลือ (8 classifiers, metric, confusion matrix) เหมือนกัน

---

## โครงสร้างการรัน Experiments

### Setup ทั่วไป
```python
root = 'Datasets'
df_cal    = load_raman_recursive('Datasets/Cal. curve (10-10-12)', 'Concentration')
df_select = load_raman_recursive('Datasets/Selectivity', 'Substance')
df_solv   = load_raman_recursive('Datasets/Solvents', 'Solvent_Type')
encode_df(df_cal, df_select, df_solv)
```

### Experiment Sweep (THC vs No THC)
รัน loop เดียวกันบน 2 dataset: Selectivity + Real-sample (Cal. curve)

| # | Feature view | สถานะใน notebook |
|---|---|---|
| A | Full Spectra (`crop_spectra` 620–1700 cm⁻¹) | commented |
| B | 9 selected peaks | commented |
| C | 8 peaks (ตัด 1222) | commented |
| D | 7 peaks (ตัด 1155) | commented |
| E | **6 peaks (ตัด 769) — final, active** | active |

### Threshold Experiments (Concentration regression → binary)
ทำ `df_cal.copy()` ทุกครั้ง — ห้ามแตะต้นฉบับ

| Threshold | Class 0 | Class 1 | สถานะ |
|---|---|---|---|
| 6.4 ppm | `≤ 6.4` (`less6.4`) | `> 6.4` (`more6.4`) | commented |
| **3.2 ppm** | `≤ 3.2` (`less3.2`) | `> 3.2` (`more3.2`) | **active** |

ทั้งสองรันด้วย 6 peaks เดียวกัน + `loovc` ครบ 8 classifiers

---

## Convention สำคัญที่ต้องจำ

1. **ห้ามใส่ `StandardScaler`** ก่อน classifier — SNV normalize มาแล้ว
2. **Pipeline ห่อแค่ classifier** — `Pipeline([('classifier', model)])`
3. **`shuffle=False`** ใน KFold — รักษาลำดับ measurement
4. **`df.copy()` ก่อน re-label threshold** — กันแตะต้นฉบับ
5. **`apply_pca` ใช้ visualization เท่านั้น** — fit ทั้งชุด, อย่าเอา PC ไปป้อนโมเดล
6. **Mask เดียวคัดทุก array** ใน `preprocessing` — กัน intensity/Y/Conc หลุด alignment
7. **`np.diff` ตรวจทิศทาง wavenumber** ก่อน reshape — ห้าม assume
8. **Permutation importance ทำหลัง CV** — fit ทั้งชุดครั้งเดียว, เร็วกว่าและเสถียรกว่ารันใน fold loop

---

## Output ที่ได้จาก 1 รอบ `loovc`

1. Progress log (ทุก ๆ 10 fold)
2. `summary_df` — DataFrame เฉลี่ย Accuracy / Precision / F1 ของ 8 โมเดล
3. Figure 1: confusion matrices grid 2×4 (สะสมทุก fold)
4. Figure 2: permutation importance grid 2×4 (vs wavenumber + top-K markers)
5. Return: `(metrics_results, summary_df, feature_importance_results, summary_raw)` สำหรับ analysis ต่อ
