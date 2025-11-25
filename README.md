# Fast API to predict Bitcoin Price

### About the project
A Machine Learning API designed to predict the next day's Bitcoin price (BTC-USD) using real-time data from Yahoo Finance and over 40 technical indicators.

### Problem
Bitcoin's extreme volatility makes manual forecasting unreliable. Traders need an automated, data-driven tool to instantly process complex market patterns and financial indicators.

### Solution
A high-performance **FastAPI** application that automates the entire inference pipeline. It fetches live market data, computes technical indicators, and utilizes a pre-trained ML model to return a precise price prediction for the next 24 hours.

### Tech stack
* **Framework:** FastAPI, Uvicorn
* **Data & ML:** Python, Pandas, NumPy, Scikit-Learn
* **Data Source:** yfinance (Yahoo Finance API)

---

### Data source

The model was trained using real, publicly available data from Yahoo Finance.

https://finance.yahoo.com/quote/BTC-USD

### Model

The model has been trained, and the model weights and scaler are in the data folder.

---

## Getting Started

Follow these instructions to set up a local copy of the project.

### Prerequisites

In this project I used `conda` for environment management.
* Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Installation

1.  **Clone the repository** (or download the files to a local folder).
    ```bash
    git clone [https://github.com/ViniciusRubens/Rest_API_to_predict_Bitcoin_price](https://github.com/ViniciusRubens/Rest_API_to_predict_Bitcoin_price)
    cd your-repository-name
    ```

2.  **Create a new conda environment** (this example uses `project_env` as the name):
    ```bash
    conda create --name project_env python=3.12
    ```

3.  **Activate the new environment:**
    ```bash
    conda activate project_env
    ```

4.  **Install pip** into the environment:
    ```bash
    conda install pip
    ```

5.  **Install the required dependencies** from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

With your `project_env` environment still active, run the application using the following command:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload
```

In another terminal:

```bash
python client.py
```

## 

## What Happened (Step-by-Step)

**Terminal 2 (Client):** You ran `python client.py`.

**Client:** Your script sent a POST request to `http://localhost:3000/predict` with the JSON `{"Model": "Machine Learning"}`.

**Terminal 1 (Server):** The API, running with Uvicorn, received this request. The log `INFO: 127.0.0.1:55036 - "POST /predict HTTP/1.1" 200 OK` confirms this.

**Server (Behind the Scenes):** The moment it received this POST, the API executed the following logic within a few milliseconds:

1. It called the `data_service`.
2. The `data_service` accessed the internet and fetched the last 200 days of actual Bitcoin (BTC-USD) data using the `yfinance` library.
3. It calculated all 50+ technical analysis indicators (RSI, MACD, Bollinger Bands, etc.) based on this data.
4. It took the most recent data row (from "today"), cleaned it, and standardized it using `scaler.bin`.
5. The `prediction_service` took this "today" data and fed it into the Machine Learning model (`model.joblib`).
6. The model predicted a single number: the estimated Bitcoin price for the next day.

**Server:** The API packaged this prediction and the last known price into a JSON and sent them back to the client.

**Terminal 2 (Client):** Your client received this response and printed it to the terminal.

```bash
{
  "Model": "Machine Learning",
  "Last_Price": 109392.41,
  "Prediction_For_Next_Day": 105112.4
}
```

This means that:

- **Last_Price:** The last known Bitcoin price that the API fetched (today's price) was $109,392.41.
- **Prediction_For_Next_Day:** Based on all of today's technical indicators, the Machine Learning model predicts that tomorrow's price will be $105,112.40.

---

## Cleanup

To deactivate and remove the conda environment (optional).

1.  **Deactivate the environment:**
    ```bash
    conda deactivate
    ```

2.  **Remove the environment (optional):**
    ```bash
    conda remove --name project_env --all
    ```

---

## License

Distributed under the MIT License. See `LICENSE` file for more information.
