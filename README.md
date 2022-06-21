# Operational Adaptation of DNN Classifiers using Elastic Weight Consolidation

Full paper available on: https://arxiv.org/abs/2205.00147

Please cite this work as: Ghobrial, A., Zheng, X., Hond, D., Asgari, H., & Eder, K. (2022). Operational Adaptation of DNN Classifiers using Elastic Weight Consolidation. arXiv preprint arXiv:2205.00147.

The latest code available is in OvercomingCF/ContinualRetrainingV7_HyperOpt.py

This work was carried out on a computer with an i7-10750H with 16GB RAM and an NVIDIA GeForce RTX 3060 graphics card, running Ubuntu 20.04.02 LTS.

## Setup
### Instal nvidia dirvers 
Skip this step if you already have your nvidia dirvers installed. 

1. If installing a fresh ubuntu installation: 
- Install ubuntu including thrid party files from usb stick. You will be asked to add a password, this is not your computer password this some password that you will need later in seure boot. Set it to somthing simple e.g. 1234567
- And continue.

2. Enter the details of your computer
- You will then be asked to enter your computer details i.e. name, username, computer name and password. Note this password is going to be your computers password that you will use continiously so set it to something of your preference.

3. Restart

4. Perform MOK managment 
- You will be prompted to a blue screen with the heading "Perform MOK management"
- Select Enroll MOK
- Then Select Contiue
- Then it will ask you to Enrol the key(s)?
- Select yes
- Input the first password you generated in our case this was the 12345678
- Press enter then Reboot

5. After restarting log in using the second (personal password) that you generated.

6. Find the appropriate nvidia drivers version for your GPU.
- Run in a terminal: `sudo add-apt-repository ppa:graphics-drivers/ppa` then: `ubuntu-drivers devices`
- See which is the recommended.
- Alternativley use this link: https://www.nvidia.co.uk/Download/index.aspx?lang=en-uk from the Nvidia webiste to help determine the appropriate drivers for your machine. 

7. Install nvidia drivers by running. Note you may need to adjust the nvidia drivers version found from the previous step. 
- Run in a terminal: `sudo apt install nvidia-driver-470` or run: `sudo ubuntu-drivers autoinstall`
- You will be asked to enter your personal password and shortly after that you will asked to confirm installation by pressing "Y" followed by "Enter".

8. Reboot:
- Run in a terminal: `sudo reboot` then log in.

9. Check installation:
- Run in terminal: `nvidia-smi`
- Should come up with a box having a lot of infomration which include GPU card and Nvidia version, cuda verision etc.


10. Other notes:
- If you need to uninstall nvidia after installing it then run: sudo apt-get purge nvidia*
- If you need to install a different version of nvidia drivers after uninstalling/purging the old one then go back to Step 7 



### Install docker and nvidia-docker
Link to refer to if needed: https://docs.docker.com/engine/install/ubuntu/.
1. Uninstall old versions: `sudo apt-get remove docker docker-engine docker.io containerd runc`

2. Set up the repository: 
- Update the apt package index and install packages to allow apt to use a repository over HTTPS: 
```
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release 
```

- Add Docker’s official GPG key: `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`

- Use the following command to set up the stable repository: 
```
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`
```

### Install Pytorch

### Download datasets

### Run 
