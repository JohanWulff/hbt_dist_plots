Plot Histograms of Signal & MC for the signal/background discriminator for the CMS $HH \rightarrow \mathrm{X} \rightarrow bb\tau\tau$ Analysis.

Tested with LCG 105. 

### Setup (Lxplus9)

```
source  /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-clang16-opt/setup.sh
```

### Example

```
python plot_dists.py -m serial_baseline_param \
                     -i /eos/user/j/jowulff/res_HH/hhres_dnn_datacards/cards/flats10_test_serial_baseline_param/flats10_qcd/ \
                     -o /test_lcg105

```
