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

#### 이더넷 Hailo 사용시 다음 스크립트 참고
```
hailortcli scan --interface <인터페이스이름>
hailortcli fw-control identify --ip <보드IP>
```
# Build
```
make -j4
```

# Running
```
./detection_native <이더넷인터페이스명>
# 기입된 인터페이스명으로 이더넷 hailo 감지 후 자동으로 init 진행
```
