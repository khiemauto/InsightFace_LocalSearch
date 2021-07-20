### FACE RECOGNITION SYSTEM

File ./configs/sdk_config.yaml #select cuda, cpu...
File ./configs/cam_infos.yaml #set camera infos

Run application

```python
python main.py
```
# Fastest with cpu
```python
python main_openvino.py
```
# Install lib
```
sudo alien -i oracle-instantclient-basic-21.1.0.0.0-1.x86_64.rpm
sudo ldconfig
pip install cx_Oracle   #Oracle sql
pip install gTTS pydub  #Text to speech and play sound
```