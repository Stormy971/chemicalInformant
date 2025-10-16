# scr/iupacscraper.py

import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote

# -----------------------------
# Helper function: get SMILES from PubChem
# -----------------------------
def get_smiles(compound_name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(compound_name)}/property/IsomericSMILES/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass
    return ""

# -----------------------------
# Selenium setup (auto ChromeDriver)
# -----------------------------
options = Options()
options.add_argument("--headless")  # run in background
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# -----------------------------
# Base URL of the database
# -----------------------------
base_url = "https://srdata.nist.gov/solubility/sol_sys.aspx"
driver.get(base_url)
time.sleep(2)

all_data = []

# -----------------------------
# Step 1: Get all Series/Volumes links
# -----------------------------
series_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='srd']")
series_urls = [link.get_attribute("href") for link in series_links]
print(f"Found {len(series_urls)} series volumes")

for series_url in series_urls:
    driver.get(series_url)
    time.sleep(1)

    # -----------------------------
    # Step 2: Get all Solubility System links (compound types)
    # -----------------------------
    system_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='SolSys']")
    system_urls = [link.get_attribute("href") for link in system_links]

    for sys_url in system_urls:
        driver.get(sys_url)
        time.sleep(0.5)

        try:
            # -----------------------------
            # Step 3: Extract experimental table
            # -----------------------------
            table = driver.find_element(By.TAG_NAME, "table")
            rows = table.find_elements(By.TAG_NAME, "tr")
            headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]

            if "Concentration c1 [mol dm**-3]" not in headers:
                continue
            conc_index = headers.index("Concentration c1 [mol dm**-3]")
            temp_index = headers.index("t/°C") if "t/°C" in headers else None

            compound_name = driver.find_element(By.TAG_NAME, "h2").text.strip()
            smiles = get_smiles(compound_name)
            mol_type = compound_name.split(" with ")[0].strip()

            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) <= conc_index:
                    continue
                sol_value = cells[conc_index].text.strip()
                if not sol_value:
                    continue
                try:
                    sol_value = float(sol_value)
                    log_sol = 0 if sol_value <= 0 else pd.np.log10(sol_value)
                except:
                    continue

                temp_c = float(cells[temp_index].text.strip()) if temp_index is not None else None

                all_data.append({
                    "Compound ID": compound_name,
                    "SMILES": smiles,
                    "measured log(solubility:mol/L)": log_sol,
                    "Temperature_C": temp_c,
                    "Type": mol_type
                })

            print(f"Added {compound_name} ({len(rows)-1} rows)")

        except Exception as e:
            print(f"Error processing {sys_url}: {e}")
            continue

# -----------------------------
# Save all data
# -----------------------------
df = pd.DataFrame(all_data)
df.to_csv("data/solubilityData/iupac_nist_srd106_full.csv", index=False)
print(f"\n✅ Saved {len(df)} rows to 'iupac_nist_srd106_full.csv'")

driver.quit()
