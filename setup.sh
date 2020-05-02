





echo Hola

sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y git python3-dev python3-pip libopenmpi-dev

mkdir GitHub

cd GitHub

git clone https://github.com/SergiPonsa/Simulation_Pybullet.git
git clone https://github.com/SergiPonsa/Parameters_CrossEntrophy_TFM.git

cd ..
"""
source ~/anaconda3/etc/profile.d/conda.sh
conda init

conda activate test2
"""
pip install -r requirements.txt
