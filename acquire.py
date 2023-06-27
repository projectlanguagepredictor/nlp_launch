"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
#standard
import pandas as pd
import numpy as np

#scraping
import requests
from requests import get
from bs4 import BeautifulSoup

#file
import os
import json
from typing import Dict, List, Optional, Union, cast

from env import github_token, github_username

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

# --------------------------------------------------------------SELENIUM ACQUIRE------------------------------------------------------------------------------

def button_click(filename='data.json'):
    """
    This function clicks a button on a web page multiple times, extracts specific information from the page,
    and saves it to a JSON file. It returns a pandas DataFrame of the data scraped.

    Args:
        filename (str): The name of the JSON file to save the data to. Default is 'data.json'.

    Returns:
        A df containing the scraped data.
    """
    if os.path.isfile(filename):
        print('json file found and loaded')
        return pd.read_json(filename)
    else:
        print('creating df and exporting json')

    # empty lists to store the data
    click_data = []

    # create the webdriver
    driver = webdriver.Chrome()

    # access the site
    driver.get("https://github.com/topics/awesome")

    # click the button 10 times
    for _ in range(10):
        # find the button using its XPath and click it
        button = driver.find_element(By.XPATH, "//button[@class='ajax-pagination-btn btn btn-outline color-border-default f6 mt-0 width-full']")
        button.click()

        # wait for the page to load
        time.sleep(5)

    # extract the author and repo name and add them to the list
    elements = driver.find_elements(By.XPATH, '//h3[@class="f3 color-fg-muted text-normal lh-condensed"]')
    for element in elements:
        click_data.append(element.text)

    # save the data to a JSON file
    with open(filename, 'w') as f:
        json.dump({'string': click_data}, f)

    # close the driver
    driver.quit()

    return click_data

# --------------------------------------------------------------------------------------------------------------------------------------------
 
REPOS = []

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)