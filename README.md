# handwrittendigitclassifier

1. Project Goal & Overview
Goal: Develop a SECURE, web application capable of recognizing single handwritten digits (0-9) from user-uploaded images a/or #live webcam feed. See Project Spec sheet for " Phase 2: Image Acquisition & Preprocessing Pipeline "
Core Technical Objective: Implement an end-to-end system application integrating computer vision for preprocessing, a convolutional neural network for classification as well as a web framework for user interaction and API delivery.


•	Key Deliverables:
A trained and saved convulutional neural netwokr
~> saved CNN model (.h5 or SavedModel format)
~> infrastructure considerations yikes
Pushing image file to ECR.
Setting up App Runner / ECS task definition & services
Configuring IAM Roles (for ECR access potentially S3 model access)
Configuring Security Groups (inbound traffic on port, potentially only from ALB/CloudFront).
Setting up environment variables SECURELY @im_roy_lee 

A Python-based backend application (FastAPI) serving the model via API.
A web-based frontend (HTML/CSS/JavaScript/   ) for user interaction.
A Dockerfile and associated configuration for containerizing the application.
Comprehensive README :)




2. High-Level Architecture
Under Construction ---> Project sheet

Stack
Programming Language: Python 
Machine Learning: TensorFlow (>= 2.10) / Keras 
Computer Vision: OpenCV 
Numerical Operations: NumPy 
Backend: FastAPI  
(for deployment): Uvicorn 
Frontend: HTML5, CSS3, 

---> conda? 
Containerization: Docker
python-dotenv, RESTful principles, JSON data format
FastAPI's CORSMiddleware or Flask-CORS extension
