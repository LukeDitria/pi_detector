# pi_detector
Detecting and logging AI detections with a Rasberry Pi and the AI Camera or Hailo 8 accelerator

# Installing Requirements

### Update Pi if you haven't already
```commandline
sudo apt update && sudo apt full-upgrade
```

### If using Hailo accelerator
```commandline
sudo apt install hailo-all
```

### If using IMX500 (AI Camera)

```commandline
sudo apt install imx500-all
```

### Reboot Pi after install!

```commandline
sudo reboot now
```

### Install pip requirements including system-wide packages

```commandline
python -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```
