# Install Requirements

#### Install RapidJSON
```
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
mkdir build
cd build
cmake ..
sudo make install -j4
```

#### install OpenCV
```
sudo apt-get install libopencv-dev
```

#### Check

* If you want to check board connecting status of ZAiV-AHU or ZAiV-AHR, please enter as below.
```
hailortcli scan --interface [ethernet interface name]
hailortcli fw-control identify --ip [Board IP]
```
# Build
```
make -j4
```

# Running

* If you have ZAiV-AHU or ZAiV-AHR, Needs to add argument(ethernet interface name).

```
./detection_native [ethernet interface name]
```
