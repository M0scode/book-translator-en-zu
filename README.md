# English to isiZulu Book Translator

## Overview

The **English to isiZulu Book Translator** is a machine translation application that converts English text into isiZulu. It is specifically designed to assist in translating educational content and school curriculum materials, helping bridge the language gap for learners whose primary language is isiZulu.

The application provides a simple web interface where users can upload English text, translate it paragraph by paragraph, and download the translated output.

---

## Features

* Translate English text into isiZulu.
* Powered by Facebook's pre-trained **NLLB-200** multilingual translation model.
* Paragraph-to-paragraph translation that preserves document structure.
* Upload text paragraphs for translation.
* Download translated text for offline use.
* Simple and user-friendly web interface built with Streamlit.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Hugging Face Transformers
* Facebook NLLB-200 Pre-trained Model
* AutoTokenizer
* PyTorch

---

## Installation

### Prerequisites

Before installing the project, ensure you have the following installed:

* Python 3.10 or higher
* Git

### Clone the Repository

```bash
git clone https://github.com/your-username/english-to-isizulu-book-translator.git

cd english-to-isizulu-book-translator
```

### Create a Virtual Environment (Recommended)

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not available, install the main packages manually.

```bash
pip install streamlit pandas transformers torch sentencepiece
```

---

## Running the Application

Start the Streamlit application by running:

```bash
streamlit run app.py
```

Replace `app.py` with the name of your main application file if different.

The application will open automatically in your default web browser.

---

# Usage

## Step 1

Launch the application.

## Step 2

Upload a text file or paste English paragraphs into the input area.

## Step 3

Click the **Translate** button.

## Step 4

Wait while the model translates each paragraph.

## Step 5

Download the translated isiZulu text.

---

## Example

### Input

```
Education is the foundation of every successful nation.

Technology is transforming classrooms across the world.
```

### Output

```
Imfundo iyisisekelo sazo zonke izizwe eziphumelelayo.

Ubuchwepheshe buguqula amakilasi emhlabeni wonke.
```

---

# Project Structure

All project files are contained in a single repository without subfolders.

Example structure:

```text
English-to-isiZulu-Book-Translator/
│
├── app.py
├── requirements.txt
├── README.md
├── sample_input.txt
├── sample_output.txt
└── any_other_project_files.py
```

---

# Configuration

The application currently uses the Facebook **NLLB-200 Distilled 600M** model.

```python
model_name = "facebook/nllb-200-distilled-600M"
```

You can change the model by replacing the model name with another compatible Hugging Face translation model.

Other configurable options include:

* Input text source
* Uploaded file format
* Translation language codes
* Batch size (if implemented)
* Maximum sequence length

---

# Model Information

This project uses Facebook AI's multilingual translation model:

* **Model:** `facebook/nllb-200-distilled-600M`
* Supports over 200 languages.
* Loaded using the Hugging Face Transformers library.
* Automatically downloads the model the first time it is used.

---

# Troubleshooting

## Model downloads slowly

The first run downloads several gigabytes of model files.

Solution:

* Ensure a stable internet connection.
* Wait until the download completes.

---

## Out of Memory Error

Large translation models require significant RAM.

Solution:

* Close unnecessary applications.
* Translate smaller batches of text.
* Use a machine with more memory.

---

## Streamlit does not start

Verify Streamlit is installed.

```bash
pip install streamlit
```

---

## ModuleNotFoundError

Install missing packages.

```bash
pip install -r requirements.txt
```

---

## Translation is slow

The first translation is usually slower because the model is loading.

Subsequent translations will be significantly faster.

---

# Future Improvements

Potential enhancements include:

* Full document translation (PDF, DOCX)
* Book chapter translation
* Translation memory
* Side-by-side bilingual view
* Custom English–isiZulu translation model
* Offline translation support
* GPU acceleration
* Improved formatting preservation
* Export to PDF and Word documents

---

# Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature/new-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature/new-feature
```

5. Open a Pull Request describing your changes.

Please ensure that your code follows Python best practices and includes appropriate documentation where necessary.

---

# License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software in accordance with the terms of the MIT License.

---

# Acknowledgements

This project makes use of the following open-source technologies:

* Facebook AI Research (FAIR)
* Hugging Face Transformers
* Streamlit
* Pandas
* PyTorch

Special thanks to the open-source community for providing the tools that make multilingual machine translation accessible.

---

## Author

Developed as part of a project to improve access to education by translating English school curriculum materials into isiZulu, helping bridge language barriers for learners.
