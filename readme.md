# Workflow
1. Connect the Hub to your laptop using bluetooth
2. Go to the ```/dev``` folder
3. Find the correct ttys device. You can do this using ```cat ttysFile```. If this outputs a bunch of random number stuff you have the correct device
4. Sometimes you have to give it permissions. In that case, do ```sudo chmod 666 ttysFile```
5. Go to ``/Users/willemropke/Library/Application Support/lego-hub-tk`` or the equivalent on your laptop.
6. Open ``lego_hub.yaml``
7. Uncomment the configuration you want to use. I use serial with a specified port. Add here the serial device from before to ```device:```
8. Save the file
9. In the cropbot project, go into the ```lego-hub-tk-main``` folder.
10. Run ```python run_command.py cp ../driver.py 1``` to copy the driver code to cropbot. This is only necessary when you have updated this code
11. Sometimes you have to run a command, then interrupt the command from the keyboard and then run it again to make it work...
12. Run ```python run_command.py start 1``` to start the driver.