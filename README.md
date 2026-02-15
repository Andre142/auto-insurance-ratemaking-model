# Auto Insurance Ratemaking Model

**Objective:** Calculate the "Pure Premium" (fair price) for a portfolio of 10,000 drivers using Generalized Linear Models (GLMs).

## Project Overview
This project simulates a standard actuarial ratemaking workflow. It separates risks into **Frequency** (how often accidents happen) and **Severity** (how much they cost).

By utilizing **Python (Statsmodels)**, I built a predictive model that segments drivers based on age and vehicle value, identifying key risk cohorts to inform pricing strategy.

* **Frequency Modeling:** Used a **Poisson Distribution** to model claim countss.
* **Severity Modeling:** Used a **Gamma Distribution** to model claim costs.
* **Pure Premium Calculation:** Implemented the insurance pricing formula:
    $$Pure\ Premium = Frequency \times Severity$$
* **Risk Segmentation:** Validated the 'Young Driver Surcharge' by proving that age is a major predictor of accident frequency and total cost.

## Key Results & Visualization
The model successfully reproduced real-world actuarial phenomena.

* **Age Segmentation:** The output reveals a sharp "risk cliff" at **age 25**, where the calculated pure premium drops significantly (approx. 2.5x reduction).
* **Statistical Significance:** The GLM summary confirms that `Vehicle_Value` is a statistically significant predictor ($P < 0.001$) for claim severity.


<img width="581" height="472" alt="Screenshot 2026-02-15 021340" src="https://github.com/user-attachments/assets/7ffc9870-c5fe-4878-8acf-1baa5f2c63f9" />

* **Language:** Python
* **Libraries:**
    * `pandas` 
    * `statsmodels` 
    * `numpy` 
    * `matplotlib` 

## How to Run
1.  Clone the repository:
    ```bash
    git clone 
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy statsmodels matplotlib
    ```
3.  Run the model:
    ```bash
    python ratemaking_model.py
    ```

## Sample Output
```text
--- FREQUENCY MODEL SUMMARY (Poisson) ---
const           -3.52   (P>|z| = 0.000)
age_Young        0.48   (P>|z| = 0.000)

--- SAMPLE RATEMAKING OUTPUT ---
   driver_age  pred_frequency  pred_severity  pure_premium
0          56        0.068945    2819.539700    194.393388
1          69        0.051238    2801.475396    143.542829


