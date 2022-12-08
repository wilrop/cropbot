# CropBot
CropBot is a demo of several techniques in AI including statistical modelling, multi-objective optimization, planning and preference learning. It is written for the LEGO Mindstorms platform. 

The CropBot project consists of three main components. The first component is the driver and is meant to be run on the LEGO Mindstorms hub. This component's only responsibility is executing the movement commands and taking images of the terrain.

The second component is the CropBot logic itself. Here, an algorithm is executed which learns a model of the underlying terrain and, given the preferences of the user, composes a monitoring schedule. 

The final component is the interactive dashboard. From this dashboard, a user can determine their preferences over the objectives and get a live visualisation of the monitoring progress.


## Citation
CropBot was presented as a demo at the BNAIC/BeNeLearn 2022 conference, held in Mechelen Belgium. If you would like to cite this demo, please use the following bibtex snippet.
````bibtex
@inproceedings{ropke2022multi,
  title={Multi-Objective Scheduling for Agricultural Interventions},
  author={R{\"o}pke, Willem and Pollaci, Samuele and Vandenbogaerde, Bram and Li, Jiahong and Coppens, Youri},
  booktitle={BNAIC/BENELEARN},
  year={2022}
}
````

## Setup
Unfortunately, the setup for CropBot is non-trivial and requires executing several steps. It is important that you first download and install the following project: [https://github.com/smr99/lego-hub-tk](https://github.com/smr99/lego-hub-tk). This software is necessary to communicate efficiently with the CropBot. Once this is done, execute the following steps in the correct order to get CropBot up and running.


1. Connect the Hub to your laptop using bluetooth.
2. Go to the ```/dev``` folder.
3. Find the correct ttys device. You can do this using ```cat ttysFile```. If this outputs a bunch of random number stuff you have the correct device.
4. Sometimes you have to give it permissions. In that case, do ```sudo chmod 666 ttysFile```.
5. Go to ``/Users/username/Library/Application Support/lego-hub-tk`` or the equivalent on your laptop.
6. Open ``lego_hub.yaml``.
7. Uncomment the configuration you want to use. I use serial with a specified port. Add here the serial device from before to ```device:```.
8. Save the file.
9. In the cropbot project, go into the ```lego-hub-tk-main``` folder.
10. Run ```python run_command.py cp ../driver.py 1``` to copy the driver code to cropbot. This is only necessary when you have updated this code.
11. Sometimes you have to run a command, then interrupt the command from the keyboard and then run it again to make it work...
12. Run ```python run_command.py start 1``` to start the driver.