# FPGA-CNN-Wafer-Optimization

## FPGA-Based CNN Accelerator for Semiconductor Wafer Defect Detection

Software-side research and experimentation for optimizing CNN inference for semiconductor wafer-map defect classification, with future FPGA implementation planned on the PYNQ-Z2.

## Dataset

WM-811K / LSWMD dataset.

The dataset is stored at:

`data/raw/LSWMD.pkl`

The dataset is tracked using Git LFS.

### Dataset used

- Original records: 811,457
- Labeled training samples: 54,355
- Labeled test samples: 118,595
- Input size: 1 × 48 × 48
- Classes: 9

Classes:

```text
none
Center
Edge-Loc
Edge-Ring
Loc
Scratch
Random
Donut
Near-full
Baseline CNN

Architecture:

Input 1×48×48
     ↓
Conv 3×3: 1→16
     ↓
ReLU + MaxPool
     ↓
Conv 3×3: 16→32
     ↓
ReLU + MaxPool
     ↓
Conv 3×3: 32→64
     ↓
ReLU + MaxPool
     ↓
Fully Connected
     ↓
9-class output

Parameters:

319,497

Results:

Accuracy : 93.7797%
Macro F1 : 59.2723%
Method 1 — Reduced Numerical Bit-Width

Intermediate CNN numerical representations were quantized at different bit widths.

Configuration	Accuracy	Macro F1
FP32	93.7797%	59.2723%
8-bit	93.8522%	58.9188%
7-bit	92.8926%	57.9344%
6-bit	91.3023%	57.2462%
5-bit	89.3132%	46.6567%
4-bit	51.8976%	12.4377%

The 8-bit configuration preserved the baseline accuracy in the tested experiment.

Method 2 — Zero-Skipping

Activation sparsity was analyzed to estimate the convolution MAC operations that could potentially be skipped.

Overall potential MAC reduction: 56.3581%

Layer results:

Layer	Zero Rate	Potential MAC Skip
features.1	38.6157%	37.3890%
features.4	59.3862%	56.1693%
features.7	84.4261%	75.5159%

This is a theoretical arithmetic reduction. A hardware implementation must actually avoid zero operations to obtain a corresponding latency benefit.

Method 3 — Depthwise-Separable Convolution

Standard 3×3 convolutions were replaced with depthwise 3×3 followed by pointwise 1×1 convolution.

Results:

Parameters                    : 299,379
Standard convolution MACs    : 5,640,192
DSC convolution MACs         : 771,840
Convolution MAC reduction    : 86.32%

Accuracy                     : 94.9517%
Macro F1                     : 56.9926%
CPU latency                  : 0.315361 ms/image
Combined Preliminary Experiment

The preliminary combined experiment used:

Depthwise-separable convolution
+
8-bit fake quantization
+
Zero-activation analysis
+
CPU parallelism

Results:

Accuracy : 94.7898%
Macro F1 : 57.4793%

CPU measurements:

1 thread : 0.792915 ms/image
2 threads: 0.659841 ms/image
4 threads: 0.521177 ms/image

The 1-to-4 thread measurement corresponds to approximately 34.3% lower latency within that particular benchmark.

The combined result is preliminary. A common standardized benchmark is required before making a final baseline-versus-optimized latency claim.

Overall Results
Experiment	Accuracy	Macro F1
Baseline	93.7797%	59.2723%
Method 1 — 8-bit	93.8522%	58.9188%
Method 2 — Zero-Skipping	93.7797%	59.2723%
Method 3 — DSC	94.9517%	56.9926%
Combined Preliminary	94.7898%	57.4793%
Repository Structure
FPGA-CNN-Wafer-Optimization/
├── data/
│   ├── raw/
│   │   └── LSWMD.pkl
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── rtl/
├── src/
├── tb/
├── weights/
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
Current Status
[✓] WM-811K dataset loading
[✓] Dataset analysis
[✓] Train/test split
[✓] 48×48 input selection
[✓] Baseline CNN
[✓] Reduced-bit experiment
[✓] Zero-skipping analysis
[✓] Depthwise-separable convolution
[✓] Combined preliminary experiment
[ ] Standardized final latency benchmark
[ ] Final optimized CNN
[ ] RTL accelerator
[ ] RTL verification
[ ] Vivado synthesis
[ ] PYNQ-Z2 implementation
Future FPGA Work

The planned hardware accelerator will investigate:

Quantized CNN inference
Depthwise-separable convolution
Zero-skipping
Parallel MAC architecture
BRAM-based buffering
RTL verification against the software model
Vivado synthesis and implementation
PYNQ-Z2 deployment

Future hardware metrics will include LUTs, FFs, DSPs, BRAMs, Fmax, cycles/image, FPGA latency, throughput and power.
