import requests
import os
from urllib.parse import urlsplit, unquote
import pandas as pd


'''
Given your name/organization and an email address and the ticker name of the company, will do a get request for that company's respective cik number
'''
def _get_CIK_from_ticker(username,email_address,ticker):
    user_agent_string = username + " " + email_address
    headers = {
    'User-Agent': user_agent_string,
    'Accept-Encoding': 'gzip, deflate' 
    }

    ticker_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(ticker_url, headers=headers)
    ticker_data = response.json()

    cik = None

    for company in ticker_data.values():
        if company['ticker'] == ticker:
            cik = str(company['cik_str']).zfill(10)
            break

    return str(cik)

def _get_10Q_filings(username,email_address,cik):
    user_agent_string = username + " " + email_address
    headers = {
        'User-Agent':user_agent_string,
        'Accept-Encoding': 'gzip, deflate'
    }

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(submissions_url, headers=headers)
    submissions_data = response.json()

    
    recent_filings = submissions_data['filings']['recent'] # we should porlly change this

    ten_q_filings = []
    for i in range(len(recent_filings['form'])):
        if recent_filings['form'][i] == '10-Q':
            ten_q_filings.append({
                'accessionNumber': recent_filings['accessionNumber'][i],
                'filingDate': recent_filings['filingDate'][i],
                'reportDate': recent_filings['reportDate'][i],
                'primaryDocument': recent_filings['primaryDocument'][i]
            })

    # Show the most recent 10-Q
    print(f"Found {len(ten_q_filings)} recent 10-Q filings.")
    return ten_q_filings


def get_10Q_filings_from_ticker(username,email_address,ticker):
    cik = _get_CIK_from_ticker(username,email_address,ticker)
    return _get_10Q_filings(username,email_address,cik),cik
#print(_get_CIK_from_ticker("Lawrence Xie","xie.law@northeastern.edu","AAPL"))
#print(get_10Q_filings_from_ticker("Lawrence Xie","xie.law@northeastern.edu","AAPL")[0])

def get_10Q_from_filings(cik,accession,primary_doc):
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession.replace("-","")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"
    return url

def download_from_url(username,email_address,url, dest_dir=".", filename=None, headers=None):
    user_agent_string = username + " " + email_address
    headers = headers or {
        "User-Agent": user_agent_string,
        "Accept-Encoding": "gzip, deflate"
    }
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()

    if filename is None:
        path = unquote(urlsplit(url).path)
        filename = os.path.basename(path) or "download"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest_path
#print(get_10Q_from_filings("0000320193","0000320193-26-000013","aapl-20260328.htm"))

#download_from_url("Lawrence Xie","xie.law@northeastern.edu","https://www.sec.gov/Archives/edgar/data/320193/000032019326000013/aapl-20260328.htm")

def download_from_ticker(username,email_address,ticker,limit=10):
    #cik = _get_CIK_from_ticker(username,email_address,ticker)
    Q_filings,cik = get_10Q_filings_from_ticker(username,email_address,ticker)
    #Q_filings is a list of filing json objects
    for filing in Q_filings[:limit]:
        Q_form_url = get_10Q_from_filings(cik,filing["accessionNumber"],filing['primaryDocument'])
        download_from_url(username,email_address,Q_form_url)

#download_from_ticker("Lawrence Xie","xie.law@northeastern.edu","AAPL",3)


# # Open and parse all tables in the HTML file
# file_path = "aapl-20250628.htm"
# all_tables = pd.read_html(file_path)

# # Look through the list of dataframes found to locate the financial sheets
# print(f"Found {len(all_tables)} tables in this document.")

# # View the first table found
# #print(all_tables)
# print(all_tables[4:11])
# print(all_tables[13:16])
# print(all_tables[17:30])
# #no 11 12
# #yes 13 14 15 17 18 19 20

def access_htm(filepath):
    all_tables = pd.read_html(filepath)
    return all_tables[4:11] + all_tables[13:16] + all_tables[17:30]
print(access_htm("aapl-20250628.htm"))