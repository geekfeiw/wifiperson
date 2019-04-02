# Record WiFi with Unix Time-stamp 

1. Follow [Linux CSI Tool Installation Instructions](http://dhalperi.github.io/linux-80211n-csitool/installation.html)

2. Before run the 
```
4. Build the Userspace Logging Tool

Build log_to_file, a command line tool that writes CSI obtained via the driver to a file:

make -C linux-80211n-csitool-supplementary/netlink
```
put **log_to_file_time.c** and **Makefile** into the folder **netlink** first. 


## How to use: with log_to_file_time.c, one can log time-stamps
 
 ```
 sudo ../netlink/log_to_file_time ~/Desktop/log.dat ~/Desktop/time.txt 
 
````
