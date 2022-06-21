# Operational Adaptation of DNN Classifiers using Elastic Weight Consolidation

Full paper available on: https://arxiv.org/abs/2205.00147

Please cite this work as: Ghobrial, A., Zheng, X., Hond, D., Asgari, H., & Eder, K. (2022). Operational Adaptation of DNN Classifiers using Elastic Weight Consolidation. arXiv preprint arXiv:2205.00147.

The latest code available is in OvercomingCF/ContinualRetrainingV7_HyperOpt.py

This work was carried out on a computer with an i7-10750H with 16GB RAM and an NVIDIA GeForce RTX 3060 graphics card, running Ubuntu 20.04.02 LTS.

## Machine Setup
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
3. Install Docker Engine:
- Update the apt package index, and install the latest version of Docker Engine and containerd, or go to the next step to install a specific version: 
```
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io
```

- Verify that Docker Engine is installed correctly by running the hello-world image: `sudo docker run hello-world`

- Ensure Docker can run without sudo (if `$USER` below doesn't work just replace it with your username):
```
sudo usermod -aG docker $USER

su - $USER

id -nG
```

4. Install NVIDIA Docker (run in terminal the following commands):
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -`

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)`

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list`

sudo apt-get update`

sudo apt-get install nvidia-docker2`

sudo pkill -SIGHUP dockerd`
```

5. Test environment and to make sure everything is installed correctly: `sudo docker run --runtime=nvidia --rm nvidia/cuda:11.0-base nvidia-smi`

### Instal Pytorch docker and setup working container
1. Download Pytorch (may take a few minutes), in terminal run `docker pull pytorch/pytorch:latest`.

2. Create a docker container from the pytorch image, start it and have a docker volume with it: `docker run -t --runtime=nvidia -d --name [container name] -v [path on host]:[path on container] -p 5000:80 [Image name]`. For example: `docker run -t --runtime=nvidia -d --name Operation_Adaptation_DNN_EWC -v ~/gits:/home -p 5000:80 pytorch/pytorch:latest`
- Useful link on volumes: https://www.digitalocean.com/community/tutorials/how-to-share-data-between-the-docker-container-and-the-host

3. Work on docker container through terminal: `docker exec -it [container name] bash`. For example `docker exec -it Operation_Adaptation_DNN_EWC bash`


Now your machine should be setup to run the operational adaptation.

## Run Operational Adaptation experiments
### Clone this repositry 
1. Open a new terminal 

2. Make new directory called gits: `mkdir ~/gits` 

3. Enter this directory: `cd ~/gits/`

4. Clone this repository: `git clone https://github.com/Abanoub-G/EWC_PoL_Adaptation.git`

### Run experiments  
1. Inside the docker container created earlier navigate to where the repository file is.

2. Go to the OvercomingCF folder `cd OvercomingCF`

3. Run `python3 ContinualRetrainingV7_HyperOpt.py`. This will automatically download the datasets used in the experiment if the don't already exist and will run the experiments.
 
