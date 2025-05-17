#!/bin/bash  
echo "Creating virtual environment..."  
# Exit immediately if a command exits with a non-zero status.  
set -e  

# Create a virtual environment named "Deep Learning For Image Analyses"  
echo "Creating virtual environment 'Deep_Learning_for_Sequence_Analysis'..."  
python3 -m venv "Deep_Learning_for_Sequence_Analysis"  

# Activate the virtual environment  
echo "Activating virtual environment..."  
source "Deep_Learning_for_Sequence_Analysis/bin/activate"  

# Install requirements  
echo "Installing requirements from requirements.txt..."  
pip install --upgrade pip  
pip install -r requirements.txt  

echo "Virtual environment 'Deep_Learning_for_Sequence_Analysis' created and requirements installed!"  
echo "Remember to activate the environment before running your scripts: source 'Deep_Learning_for_Sequence_Analysis/bin/activate'" 