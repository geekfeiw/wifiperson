# Recornd WiFi with Unix Time-stamp 

1. Follow [Linux CSI Tool Installation Instructions](http://dhalperi.github.io/linux-80211n-csitool/installation.html)

2. Before run the 
```
4. Build the Userspace Logging Tool

Build log_to_file, a command line tool that writes CSI obtained via the driver to a file:

make -C linux-80211n-csitool-supplementary/netlink
```
put *log_to_file_time.c* and *Makefile* into the folder *netlink* first.




##How to use:
    
> 
sudo airmon-ng check kill

sudo service network-manager stop


    receiver:
	./setup_monitor_csi.sh 64 HT20
	sudo ../netlink/log_to_file ~/Desktop/log.dat	
   ### new operation, with log_to_file_time.c, one can log time-stamps
        sudo ../netlink/log_to_file_time ~/Desktop/log.dat ~/Desktop/time.txt 

    transmitter:
	./setup_inject.sh 64 HT20
	echo 0x4101 | sudo tee `sudo find /sys -name monitor_tx_rate`
	sudo ./random_packets 1000000000 100 1

Please check the source code for random_packets.c to understand the meaning of
these parameters and the other parameters that are available.
