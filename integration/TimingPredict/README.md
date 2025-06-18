# Timing Predict intergration

## Usage

1. **Clone the TimingPredict repository**

 ```bash
   git clone https://github.com/TimingPredict/TimingPredict.git
   cd TimingPredict
   # follow the original repo’s INSTALL / README instructions here
```
   

2. Copy the integration assets

From your CirSTAG checkout (or wherever the files live):
```bash
    cp -r path/to/CirSTAG/cirstag/my_utils ./my_utils
    cp  path/to/CirSTAG/integration/TimingPredict/run_me.py  .
```

3. Run the experiment script

```bash
    python3 run_me.py
```
