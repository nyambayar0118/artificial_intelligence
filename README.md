# Artifical Intelligence class models

This is a github repository for sharing and storing the files for the class "Artificial intelligence".

---

## 📁 Repository Structure

This repository contains three main folders:

```text
root/
├── lab_6/              --> Contains the perceptron model
├── lab_7/              --> Contains the Naive Bayes spam mail classifier model
├── lab_8/              --> Contains the Decision tree loan approver model
├── .gitignore
├── LICENSE
└── README.md
```

### 🔹 lab_6 Perceptron Model

```text
lab_6/
├── Iris.csv            --> Data of Iris type flowers
├── note.ipynb
└── perceptron.py       --> Contains code for Binary, Dual, Multi Class perceptron
```

### 🔹 lab_7 Naive Bayes Model

```text
lab_7/
├── data/               --> Data of the spam and ham emails stored in separate files
├── spam_data_mac/      --> Data for mac users
├── details_ham.txt     --> Word count matrix of each ham mail
├── details_spam.txt    --> Word count matrix of each spam mail
├── ham_unigram.txt     --> Unigram of the ham mails
├── model.ipynb
├── model.py            --> The Naive Bayes model code
└── spam_unigram.txt    --> Unigram of the spam mails
```

### 🔹 lab_8 Decision Tree Model

```text
lab_8/
├── data/               --> Data of the people who had requested a loan
├── model.py            --> The Decision Tree creator model
└── test.ipynb
```

---

## 🚀 Getting Started

### Prerequisites

- Python (version 3.12 or higher)

### Installation

```bash
git clone https://github.com/nyambayar0118/artificial_intelligence.git
cd <project-folder>
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn chardet
```

▶️ Running the Project (Example for lab_6)

```bash
cd ./lab_6
python perceptron.py
```

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Nyambayar.O

GitHub: https://github.com/nyambayar0118

Email: nyambayar2014@gmail.com
