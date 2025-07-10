# CMCS

This project is a Mesa implementation of the NetLogo Language Change model:
[NetLogo Language Change model](https://www.netlogoweb.org/launch#https://www.netlogoweb.org/assets/modelslib/Sample%20Models/Social%20Science/Language%20Change.nlogo)

This project uses [Mesa](https://mesa.readthedocs.io/) to run interactive language change model applications with different grammar configurations.

## Setup

1. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Apps

- **2 Grammars:**  
  Run the main app with two grammars:
  ```bash
  solara run app.py
  ```

- **3 Grammars:**  
  Run the app with three grammars:
  ```bash
  solara run 3_app.py
  ```

## Running the experiments

- **2 Grammars:**  
  Run the main batch of experiments with two grammars:
  ```bash
  python3 batch_run.py
  ```

- **3 Grammars:**  
  Run the main batch of experiments with three grammars:
  ```bash
  python3 3_batch_run.py
  ```

---

Feel free to explore and modify the apps for your language change modeling experiments!

Original model citation:
Troutman, C. and Wilensky, U. (2007). NetLogo Language Change model. 
http://ccl.northwestern.edu/netlogo/models/LanguageChange. 
Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.