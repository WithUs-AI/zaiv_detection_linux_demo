# Install Requirements

## Install RapidJSON
```
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
mkdir build
cd build
cmake ..
sudo make install -j4
```

## install OpenCV
```
sudo apt-get install libopencv-dev
```

## 이더넷 Hailo 사용시 다음 스크립트 참고
```
hailortcli scan --interface <인터페이스이름>
hailortcli fw-control identify --ip <보드IP>
```

## 개선 사항 
* sudo ./configure_ethernet_buffers.sh <인터페이스이름> 은 코드내에서 실행
* hailo_reset_device() hailo reset api 사용됨 , 오류시 자동으로 호출해서 해결
* 이더넷 / PCIE 코드내에서 감지해 되는 인터페이스로 실행 (이더넷 우선감지 후 오류시 PCIE 코드 진행)
* **이더넷 인터페이스명은 직접 기입해야함** argument 로 지정되도록 기능추가
```
./detection_native <이더넷인터페이스명>
# 기입된 인터페이스명으로 이더넷 hailo 감지 후 자동으로 init 진행
```