# GenAI for Software Development (Predicting If Statements)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 OPTIONAL Prep](#22-optional-prep)  
* [3 Training the Model](#3-training-the-model) 
  * [3.1 Option 1: Using Google Colab](#31-option-1-using-google-colab)
  * [3.2 Option 2: Using the Command Line](#32-option-2-using-the-command-line)
  * [3.3 Some Extra Notes](#33-some-extra-notes)
* [4 Evaluating the Model](#4-evaluating-the-model)
* [5 Report](#5-report)
---

## **1. Introduction** 
This project builds a model that predicts if
statements in Python methods. The model will take as input a function containing a special token (\<MASK\>) masking a single if condition and will attempt to predict it. We use CodeT5, a pre-trained encoder-decoder Transformer model designed for code understanding and generation and fine-tune it for the purpose of predicting if statements.

---

## **2. Getting Started**  

### **2.1 Preparations** 
Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/rachelzheng03/Predicting-If-Statements.git
```

### **2.2 OPTIONAL Prep**

These preparation steps are only necessary if you would like to train the model using the command line (instead of Google Colab).

(1) Navigate into the repository:
```
~ $ cd Predicting-If-Statements
~/Predicting-If-Statements $
```

(2) Set up a virtual environment and activate it:

For macOS/Linux:
```
~/Predicting-If-Statements $ python -m venv ./venv/
~/Predicting-If-Statements $ source venv/bin/activate
(venv) ~/Predicting-If-Statements $ 
```

For Windows:
```
python -m venv venv
venv/Scripts/Activate
```

To deactivate the virtual environment, use the command:
```
(venv) $ deactivate
```
(3) Install the required dependencies:

```
(venv) ~/ $ pip install -r requirements.txt
```

## **3. Training the Model**

### **3.1 Option 1: Using Google Colab**

Run the notebook `train_model.ipynb` in Google Colab. Make sure to upload the files `ft_train_masked.csv`, `ft_valid_masked.csv`, and `ft_test_masked.csv` to the file section of Colab before running any of the cells. After the training is finished, download `final_model.zip`.

### **3.2 Option 2: Using the Command Line**
```
(venv) ~/ $ python train_model.py
```
The trained model will be saved to the folder `final_model`.

### **3.3 Some Extra Notes**

* For both options, the files are currently set to train for inputs with the \<TAB\> token. I have also provided the code to train for inputs without the \<TAB\> token, but these are commented out. This will also apply in section 4.

* The models I trained can be found at the following link: 

## **4. Evaluating the Model**
The model was evaluated on how well it predicted if statements using methods in `ft_test_masked.csv` using the following metrics: CodeBLEU, BLEU-4, and exact match.

(1) Making the predictions:

Run the notebook `get_predictions.ipynb` in Google Colab. Make sure to upload the final model as a zip file to the file section of Colab before running any of the cells and to download the .csv file at the end. The .csv file will be needed to calculate the metrics.

(2) Calculating the metrics:

Run the notebook `evaluate_model.ipynb` in Google Colab. Make sure to upload the .csv file from the previous step to the file section of Colab before running any of the cells.

## **5. Report**
The assignment report is available in the file Assignment_Report.pdf.

