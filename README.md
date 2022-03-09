<h1 align="center">
<p> Vietnamese ID Card OCR using template matching method
</h1>

# Overview
This project is just to practice my machine learning hobbies

# Note
Remember to download phobert_report.pth and transformerocr.pth and put it to ./weights folder

# Pipeline
1. Create a template data with csv file contain coordinate
2. Use doc_scanner to preprocess images
3. Matching coordinate from the template to preprocessed images
4. The OCR process is using VietOCR with TransformerOCR model and dataset to regconize vietnamese character
5. Correction process is using PhoBert dataset but my NLP knowledge is limited so I will implement this later :> 
![](./pipeline.PNG)
![](./pipeline_demo.PNG)

