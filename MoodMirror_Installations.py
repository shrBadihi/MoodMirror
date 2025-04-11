
!pip install -q google-generativeai PyMuPDF
!pip uninstall -qqy jupyterlab  # Remove unused conflicting packages
!pip install -U -q "google-genai==1.7.0"
!pip install chromadb
!pip install gradio
!pip install -q SpeechRecognition
!pip install -q pydub
!apt-get -y install ffmpeg
!pip install -q fer
!pip install -q mtcnn  # optional, for better face detection
!pip uninstall -y pillow
!pip install pillow==9.5.0
!pip install -q facenet-pytorch
