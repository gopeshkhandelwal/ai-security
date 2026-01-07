apt update && apt install python3-pip -y
pip3 install virtualenv
mkdir llm-chatbot
cd llm-chatbot
virtualenv venv
source venv/bin/activate

cat >requirements.txt <<EOF
transformers==4.48.3
torch==2.6.0
langchain==0.3.26
langchain-community==0.3.26
faiss-cpu==1.11.0
sentence-transformers==4.1.0
accelerate==1.8.1
einops==0.8.1
jinja2==3.1.6
tensorflow==2.16.1
tf-keras==2.16.0
EOF

mkdir documents
cd documents
wget -O - https://gitlab.practical-devsecops.training/-/snippets/67/raw/main/TechCorpXYZFiles.sh | bash
wget -O - https://gitlab.practical-devsecops.training/-/snippets/69/raw/main/TechCorpXYZ-DigitizationProjectData.sh | bash

cd ..
pip install -r requirements.txt

wget -O llm-chatbot.py https://gitlab.practical-devsecops.training/-/snippets/70/raw/main/llm-chatbot-sensitive-information.py
python3 llm-chatbot.py